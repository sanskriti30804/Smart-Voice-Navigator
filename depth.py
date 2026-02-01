import logging
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DistanceTest")

class DepthPipeline:
    def __init__(self):
        logger.info("Loading Models...")
        
        # 1. Load YOLOv11 (Fast object detection)
        self.yolo = YOLO("yolo11n.pt")
        
        # 2. Load ZoeDepth (Metric Depth Estimation)
        # 'Intel/zoedepth-nyu-kitti' is fine-tuned for metric (absolute) distance
        logger.info("Loading ZoeDepth (this may take a minute first time)...")
        self.depth_processor = ZoeDepthImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        self.depth_model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").eval()
        
        # Move to GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.depth_model.to(self.device)
        logger.info(f"Models loaded on {self.device}")

        logger.info(f"CUDA available: {torch.cuda.is_available()}")

    def get_object_distance(self, image_path: str, target_object: str):
        """
        Detects an object and returns its absolute distance in meters.
        """
        # Load Image
        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            return f"Error loading image: {e}"

        # --- STEP 1: Object Detection (YOLO) ---
        logger.info(f"Scanning for '{target_object}'...")
        yolo_results = self.yolo(pil_image, verbose=False)[0]
        
        found_box = None
        found_label = None
        
        # Simple string matching (In your main app, keep using your Semantic/Embedding logic!)
        # YOLO classes are simple (e.g., 'bottle', 'cup', 'person')
        for box in yolo_results.boxes:
            class_id = int(box.cls[0])
            label = yolo_results.names[class_id]
            
            # Flexible match: e.g. "bottle" matches "water bottle"
            if label.lower() in target_object.lower() or target_object.lower() in label.lower():
                found_box = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                found_label = label
                break  # Stop at first match for this test
        
        if found_box is None:
            available_objects = [yolo_results.names[int(b.cls[0])] for b in yolo_results.boxes]
            return f"Could not find '{target_object}'. Found: {set(available_objects)}"

        logger.info(f"Found '{found_label}' at box: {found_box}")

        # --- STEP 2: Depth Estimation (ZoeDepth) ---
        logger.info("Calculating metric depth...")
        inputs = self.depth_processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
            
        # Post-process to get metric depth at original image size
        post_processed = self.depth_processor.post_process_depth_estimation(
            outputs, source_sizes=[(pil_image.height, pil_image.width)]
        )
        # depth_map contains distance in meters for every pixel
        depth_map = post_processed[0]["predicted_depth"].cpu().numpy()

        # --- STEP 3: Extract Distance for the Object ---
        x1, y1, x2, y2 = map(int, found_box)
        
        # Clamp coordinates to image boundaries
        x1, x2 = max(0, x1), min(pil_image.width, x2)
        y1, y2 = max(0, y1), min(pil_image.height, y2)
        
        # Crop the depth map to the object's bounding box
        object_depth_region = depth_map[y1:y2, x1:x2]
        
        if object_depth_region.size == 0:
            return "Error: Bounding box size is zero."

        # Use MEDIAN to ignore background pixels captured in the box corners
        distance = np.median(object_depth_region)
        
        return {
            "object": found_label,
            "target_requested": target_object,
            "distance_meters": round(float(distance), 2),
            "confidence": f"{float(box.conf[0]):.2f}"
        }

# --- TEST RUN ---
if __name__ == "__main__":
    # Create the pipeline
    pipeline = DepthPipeline()
    
    # Define your test image path
    # Ensure this image actually exists in your folder!
    image_file = "data/test.jpg" 
    
    # Test 1: Search for a bottle
    result = pipeline.get_object_distance(image_file, "bottle")
    print(f"\n--- Result ---\n{result}")
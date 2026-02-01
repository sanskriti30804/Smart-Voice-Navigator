from dotenv import load_dotenv
import logging 
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions, RunContext
from livekit.plugins import (
    noise_cancellation,
    silero,
    sarvam,
    google,
    openai,
    )
from livekit.agents.llm import function_tool
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from dataclasses import dataclass, field
from typing import List, Optional, Annotated
from pydantic import Field
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer, util
import os
import json
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import torch
import numpy as np
from PIL import Image
from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class UserData:
    object_to_find: Optional[str] = None
    user_location: Optional[str] = None
    object_found: bool = False
    object_image: Optional[str] = None
    detected_box: Optional[List[float]] = None
    prev_agent: Optional[Agent] = None
    agents: dict[str, Agent] = field(default_factory=dict)

    def summarize(self) -> str:
        data = {
            "object_to_find": self.object_to_find or "unknown",
            "user_location": self.user_location or "unknown",
            "object_found": self.object_found,
            "object_image": self.object_image or "no image",
            "prev_agent": self.prev_agent.__class__.__name__ if self.prev_agent else "no previous agent"
        }
        return json.dumps(data, indent=2)

RunContext_T = RunContext[UserData]


class BaseAgent(Agent):
    async def on_enter(self, generate_reply: bool = True) -> None:
        agent_name = self.__class__.__name__
        logger.info(f"entering task {agent_name}")

        userdata: UserData = self.session.userdata
        chat_ctx = self.chat_ctx.copy()

        # add the previous agent's chat history to the current agent
        if isinstance(userdata.prev_agent, Agent):
            truncated_chat_ctx = userdata.prev_agent.chat_ctx.copy(
                exclude_instructions=True, exclude_function_call=False
            ).truncate(max_items=6)
            existing_ids = {item.id for item in chat_ctx.items}
            items_copy = [item for item in truncated_chat_ctx.items if item.id not in existing_ids]
            chat_ctx.items.extend(items_copy)

        # add an instructions including the user data as assistant message
        chat_ctx.add_message(
            role="system",  # role=system works for OpenAI's LLM and Realtime API
            content=f"You are {agent_name} agent. Current user data is {userdata.summarize()}",
        )
        await self.update_chat_ctx(chat_ctx)
        if generate_reply:
            self.session.generate_reply(tool_choice="none")

    async def _transfer_to_agent(self, name: str, context: Optional[RunContext_T] = None) -> tuple[Agent, str]:
        if context is None:
            # Called from lifecycle method, use self.session
            userdata = self.session.userdata
            current_agent = self.session.current_agent
        else:
            # Called from function tool, use provided context
            userdata = context.userdata
            current_agent = context.session.current_agent
        
        next_agent = userdata.agents[name]
        userdata.prev_agent = current_agent

        return next_agent, f"Transferring to {name}."

class Greeting(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You're a calm, real-time smart voice navigator based inside a user's house. Collect the user's target item and user's location, confirm what you heard,"
                " and let them know exactly what you're doing next. Keep responses short, actionable, and conversational."
            ),
        )
    
    @function_tool()
    async def update_object_to_find(
        self, 
        object_to_find: Annotated[str, Field(description="The object the user wants to find")], 
        context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when user provides the object they want to find"""
        userdata = context.userdata
        target = object_to_find.strip()
        userdata.object_to_find = target

        if userdata.user_location:
            next_agent, _ = await self._transfer_to_agent("object_detection", context)
            return next_agent, (
                f"Locked onto {target}. You're in {userdata.user_location}, so I'm spinning up detection now."
            )

        return "Locked onto {target}. Give me your current location so I can route you through the quickest path.".format(target=target)

    @function_tool()
    async def update_user_location(
        self, 
        user_location: Annotated[str, Field(description="The location of the user")], 
        context: RunContext_T) -> str | tuple[Agent, str]:
        """Called when user provides their location"""
        userdata = context.userdata
        location = user_location.strip()
        userdata.user_location = location

        if userdata.object_to_find:
            next_agent, _ = await self._transfer_to_agent("object_detection", context)
            return next_agent, (
                f"Copy that, you're in {location}. Scanning for {userdata.object_to_find} now."
            )

        return f"Noted—{location}. Tell me what you're hunting for and I'll spin up the right tools."

class ObjectDetectionAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You handle rapid object detection. Wait for the detection results. "
                "After detection, facilitate the next steps, offer to estimate distance if found, or suggest checking the knowledege base if not found. Listen for user's confirmation before proceeding. "
                "Do not speak until detection is complete."
            ),
        )
        # Load models once at startup to avoid re-loading latency
        logger.info("Loading YOLO and Embedding models...")
        self.yolo_model = YOLO("yolo11n.pt")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        logger.info("Models loaded.")
    
    async def on_enter(self) -> None:
        await super().on_enter(generate_reply=False)
        
        # Run detection
        message = await self._run_detection() 
        await self.session.say(message)
        
        # Enable function tool calling for user's next response
        self.session.generate_reply(tool_choice="auto")
     
    async def _run_detection(self, context: Optional[RunContext_T] = None) -> str:
        userdata = self.session.userdata if context is None else context.userdata
        target = userdata.object_to_find

        if not target:
            return "I don't know what object to look for."
        
        results = self.yolo_model(userdata.object_image)

        detected_objects = []

        for result in results:
            if result.boxes:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    name = result.names[cls_id]
                    # Convert box to simple list [x1, y1, x2, y2]
                    coords = box.xyxy[0].cpu().numpy().tolist()
                    detected_objects.append((name, coords))
        
        names_only = [obj[0] for obj in detected_objects]
        logger.info(f"Predicted Object list: {names_only}")

        if not names_only:
            userdata.object_found = False
            return f"I didn't spot any objects. Should I check the knowledge base?"

        def calculate_similarity():
            query_embedding = self.embedding_model.encode(target, convert_to_tensor=True)
            predicted_embeddings = self.embedding_model.encode(names_only, convert_to_tensor=True)
            return util.cos_sim(query_embedding, predicted_embeddings)[0]

        # similarities = await asyncio.to_thread(calculate_similarity)
        similarities = calculate_similarity() # trying without thread for now
        
        best_idx = int(similarities.argmax())
        best_match = names_only[best_idx]
        best_match_box = detected_objects[best_idx][1]
        best_score = float(similarities[best_idx])

        logger.info(f"Best match: {best_match} (Score: {best_score:.3f})")

        if best_score >= 0.5: # Adjusted threshold
            userdata.object_found = True
            userdata.detected_box = best_match_box
            logger.info(f'Object Found: {userdata.object_found}')
            return f"I found {best_match}, which looks like your {target}. Want me to estimate the distance to it?"
        
        userdata.object_found = False
        logger.info(f'Object Found: {userdata.object_found}')
        return (
            f"I see {', '.join(names_only[:3])}, but nothing matching {target}. Should I check the knowledge base?"
        )
    
    @function_tool()
    async def to_depth_estimation(self, context: RunContext_T) -> tuple[Agent, str]:
        """Call this when user wants to estimate distance to the detected object. Use when user says yes, sure, please do, go ahead, or similar confirmation."""
        logger.info("Function tool 'to_depth_estimation' called - transferring to DepthEstimationAgent")
        return await self._transfer_to_agent("depth_estimation", context)
    
    @function_tool()
    async def to_rag(self, context: RunContext_T) -> tuple[Agent, str]:
        """Call this when user wants to check the knowledge base for object location. Use when object is not found or user asks to search knowledge base."""
        logger.info("Function tool 'to_rag' called - transferring to RAGAgent")
        return await self._transfer_to_agent("rag", context)

class RAGAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You're the fallback navigator. Pull from building knowledge to guide the user to the target"
                " if detection missed it. Be definitive about directions and confirm next steps."
            ),
        )

class DepthEstimationAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You handle distance estimation to detected objects. Wait for the depth estimation results. "
                "After estimation, ask if the user needs further assistance or wants to search for another object. "
                "Do not speak until estimation is complete."
            ),
        )
        logger.info("Loading Depth Estimation model...")
        self.depth_processor = ZoeDepthImageProcessor.from_pretrained("Intel/zoedepth-nyu-kitti")
        self.depth_model = ZoeDepthForDepthEstimation.from_pretrained("Intel/zoedepth-nyu-kitti").eval()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.depth_model.to(self.device)
        logger.info(f"ZoeDepth model loaded on {self.device}")

    async def on_enter(self) -> None:
        await super().on_enter(generate_reply=False)
        
        # Run depth estimation
        message = await self._estimate_depth()
        await self.session.say(message)
        
        # Enable function tool calling for user's next response
        self.session.generate_reply(tool_choice="auto")

    async def _estimate_depth(self, context: Optional[RunContext_T] = None) -> str:
        """Run depth estimation on the detected object"""
        userdata = self.session.userdata if context is None else context.userdata
        object_to_find = userdata.object_to_find
        image_path = userdata.object_image
        box = userdata.detected_box # [x1, y1, x2, y2]

        logger.info(f"Starting depth estimation for {object_to_find}")

        if not object_to_find or not image_path or not box:
            logger.warning("Missing required data for depth estimation")
            return "I need a confirmed object detection to gauge distance."

        try:
            # 1. Load Image
            logger.info(f"Loading image from {image_path}")
            pil_image = Image.open(image_path).convert("RGB")
            logger.info(f"Image loaded: {pil_image.width}x{pil_image.height}")
            
            # 2. Generate Depth Map
            logger.info("Generating depth map...")
            inputs = self.depth_processor(images=pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.depth_model(**inputs)
            
            post_processed = self.depth_processor.post_process_depth_estimation(
                outputs, source_sizes=[(pil_image.height, pil_image.width)]
            )[0]
            depth_map = post_processed["predicted_depth"].cpu().numpy()
            logger.info(f"Depth map generated: shape {depth_map.shape}")

            # 3. Extract Distance for the specific box
            x1, y1, x2, y2 = map(int, box)
            logger.info(f"Object bounding box: [{x1}, {y1}, {x2}, {y2}]")
            
            # Clamp coordinates
            x1, x2 = max(0, x1), min(pil_image.width, x2)
            y1, y2 = max(0, y1), min(pil_image.height, y2)
            logger.info(f"Clamped box coordinates: [{x1}, {y1}, {x2}, {y2}]")
            
            object_depth_region = depth_map[y1:y2, x1:x2]
            
            if object_depth_region.size == 0:
                logger.warning("Empty depth region after cropping")
                return f"I see the {object_to_find}, but I can't get a clear depth reading."

            # Calculate Median Distance
            distance = np.median(object_depth_region)
            distance_str = f"{distance:.2f}"
            logger.info(f"Calculated distance: {distance_str} meters")

            return f"The {object_to_find} is about {distance_str} meters away. Close in carefully. Need anything else?"

        except Exception as e:
            logger.error(f"Depth estimation error: {e}", exc_info=True)
            return "I'm having trouble reading the depth sensors right now."
    
    @function_tool()
    async def search_new_object(self, context: RunContext_T) -> tuple[Agent, str]:
        """Call this when user wants to search for a different object. Use when user says they want to find something else or start a new search."""
        logger.info("Function tool 'search_new_object' called - resetting and transferring to Greeting")
        # Reset userdata for new search
        context.userdata.object_to_find = None
        context.userdata.user_location = None
        context.userdata.object_found = False
        return await self._transfer_to_agent("greeter", context)
    
    @function_tool()
    async def end_session(self, context: RunContext_T) -> str:
        """Call this when user wants to end the session or says goodbye, thank you, that's all, I'm done, or similar."""
        logger.info("Function tool 'end_session' called - ending navigation")
        return "Navigation complete. Have a great day!"


async def entrypoint(ctx: agents.JobContext):

    userdata = UserData()

    cwd = os.getcwd()
    filepath = os.path.join(cwd, "data/test.jpg") # in real word case this image will be taken from somewhere like S3 
    userdata.object_image = filepath

    userdata.agents.update(
        {
            "greeter": Greeting(),
            "object_detection": ObjectDetectionAgent(),
            "rag": RAGAgent(),
            "depth_estimation": DepthEstimationAgent(),
        }
    )

    session = AgentSession(
        stt = openai.STT(
        model="gpt-4o-transcribe",
        language="en",
        ),  
        llm=google.LLM(model="gemini-3-flash-preview"),
        tts=sarvam.TTS(
            target_language_code="en-IN",
            speaker="hitesh"
        ),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        userdata=userdata
    )

    await session.start(
        room=ctx.room,
        agent=userdata.agents["greeter"],
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

# ----------------------------
# Minimal health server
# ----------------------------
class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/healthz":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
        else:
            self.send_response(404)
            self.end_headers()


def start_health_server():
    http_server = HTTPServer(("0.0.0.0", 4000), HealthHandler)
    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()

if __name__ == "__main__":  
    # Start lightweight health server
    start_health_server()

    # Boot LiveKit agent
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
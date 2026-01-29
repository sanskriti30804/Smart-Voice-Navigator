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
from typing import Optional, Annotated
from pydantic import Field
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer, util
import os
import json
import asyncio
logger = logging.getLogger(__name__)

load_dotenv()

@dataclass
class UserData:
    object_to_find: Optional[str] = None
    user_location: Optional[str] = None
    object_found: bool = False
    object_image: Optional[str] = None
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
    async def on_enter(self) -> None:
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
                "You handle rapid object detection. Use the latest image and target name from userdata"
            ),
        )
        # Load models once at startup to avoid re-loading latency
        logger.info("Loading YOLO and Embedding models...")
        self.yolo_model = YOLO("yolo11n.pt")
        self.embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        logger.info("Models loaded.")
    
    async def on_enter(self) -> None:
        await super().on_enter()
        
        # Run detection
        message = await self._run_detection() 
        await self.session.say(message)
     
    async def _run_detection(self, context: Optional[RunContext_T] = None) -> str:
        userdata = self.session.userdata if context is None else context.userdata
        target = userdata.object_to_find

        if not target:
            return "I don't know what object to look for."
        
        def run_yolo():
            return self.yolo_model(userdata.object_image)

        results = await asyncio.to_thread(run_yolo)

        names = []
        for result in results:
            if result.boxes:
                # Extract class names safely
                names = [result.names[int(cls)] for cls in result.boxes.cls.tolist()]
        
        logger.info(f"Predicted Object list: {names}")

        if not names:
            userdata.object_found = False
            return f"I didn't spot any objects. Should I check the knowledge base?"

        def calculate_similarity():
            query_embedding = self.embedding_model.encode(target, convert_to_tensor=True)
            predicted_embeddings = self.embedding_model.encode(names, convert_to_tensor=True)
            return util.cos_sim(query_embedding, predicted_embeddings)[0]

        similarities = await asyncio.to_thread(calculate_similarity)
        
        best_idx = int(similarities.argmax())
        best_match = names[best_idx]
        best_score = float(similarities[best_idx])

        logger.info(f"Best match: {best_match} (Score: {best_score:.3f})")

        if best_score >= 0.5: # Adjusted threshold
            userdata.object_found = True
            return (
                f"I found {best_match}, which looks like your {target}. Want me to estimate the distance?"
            )
        
        userdata.object_found = False
        return (
            f"I see {', '.join(names[:3])}, but nothing matching {target}. Should I check the knowledge base?"
        )

    @function_tool()
    async def detect_object(self, context: RunContext_T) -> tuple[Agent, str]:
        return await self._run_detection(context)
    
    @function_tool()
    async def to_depth_estimation(self, context: RunContext_T) -> tuple[Agent, str]:
        return await self._transfer_to_agent("depth_estimation", context)
    
    @function_tool()
    async def to_rag(self, context: RunContext_T) -> tuple[Agent, str]:
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
                "You estimate distance to the detected object. Use userdata for the target and latest image"
            ),
        )
    async def on_enter(self) -> None:
        await super().on_enter()
        message = await self.estimate_depth()
        await self.session.say(message)


    async def estimate_depth(self, context: Optional[RunContext_T] = None) -> str:
        """Called when user wants to estimate the distance to the object"""
        userdata = userdata = self.session.userdata if context is None else context.userdata
        object_to_find = userdata.object_to_find
        image = userdata.object_image
        predicted_depth = 10
        return f"{object_to_find} is about {predicted_depth} meters ahead. Close in carefully." if object_to_find else "Can't gauge distance without a target."


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

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
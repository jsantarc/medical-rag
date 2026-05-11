# main.py
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse

from schemas import ChatRequest
from agent import stream_agent_response

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from deps import get_vectorstore
    get_vectorstore()
    logger.info("Vectorstore warmed up")
    yield

app = FastAPI(
    title="Medical RAG Chatbot",
    description="Ask clinical questions answered from your medical PDF library.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "index.html")


@app.post("/chat", tags=["chat"])
async def chat(request: ChatRequest):
    """Stream a response to a medical question."""
    logger.info(f"Question: {request.message}")

    async def token_generator():
        async for token in stream_agent_response(message=request.message):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")

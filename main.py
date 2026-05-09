# main.py
import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from schemas import ChatRequest

from fastapi.responses import StreamingResponse, FileResponse
from agent import stream_agent_response, reset_agent

load_dotenv()

# logging lets you see what's happening in the terminal
# INFO level shows requests, warnings, and errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app — this is the object that holds all your routes
# title and description show up in the /docs UI automatically
app = FastAPI(
    title="Medical RAG Chatbot",
    description="Ask clinical questions answered from your medical PDF library.",
    version="1.0.0",
)

# CORS allows browser clients to call the API from a different domain
# e.g. your React frontend on localhost:3000 calling this API on localhost:8000
# allow_origins=["*"] means ANY domain can call it — tighten in production
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

    logger.info(f"[{request.session_id}] Question: {request.message}")

    async def token_generator():
        async for token in stream_agent_response(
            message=request.message,
            session_id=request.session_id
        ):
            yield token

    return StreamingResponse(
        token_generator(),
        media_type="text/plain"
    )


@app.post("/reset")
async def reset_memory():
    reset_agent()
    if os.path.exists("memory.db"):
        os.remove("memory.db")
    return {"status": "memory cleared"}
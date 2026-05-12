import os
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings



load_dotenv()

CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHROMA_DIR  = str(Path(__file__).parent / "embeddings")

@lru_cache(maxsize=1)
def make_obj():
       # streaming=True is required for token-by-token streaming later
    # temperature=0 keeps medical answers deterministic
    return ChatOpenAI(model=CHAT_MODEL, temperature=0, streaming=True)

@lru_cache(maxsize=1)
def get_vectorstore():
    if not os.path.exists(CHROMA_DIR):
        raise RuntimeError(
            f"Vectorstore not found at '{CHROMA_DIR}/'. "
            "Run python ingest.py first."
        )
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
    )


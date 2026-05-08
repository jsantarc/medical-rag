# schemas.py
from pydantic import BaseModel, Field

class ChatRequest(BaseModel):
    """What the client sends in."""
    
    message: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="The user's medical question."
    )
    session_id: str = Field(
        default="default",
        description="Identifies the conversation session."
    )

class GroundingSource(BaseModel):
    """A single source document that backs up the answer."""
    
    source: str = Field(description="The filename the chunk came from.")
    quote: str = Field(description="The exact passage retrieved from the document.")
    score: float = Field(description="Relevance score — lower is better.")

class ChatResponse(BaseModel):
    """What the API sends back."""
    
    answer: str = Field(description="The agent's answer to the question.")
    grounding: list[GroundingSource] = Field(
        default=[],
        description="The source passages that the answer is based on."
    )


#
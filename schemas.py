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


# test_schemas.py
from schemas import ChatRequest, ChatResponse, GroundingSource

# test a valid request
request = ChatRequest(
    message="What is the dosage for metformin?",
    session_id="user-42"
)
print("Request:", request)

# test default session_id
request_no_session = ChatRequest(
    message="What are the side effects of aspirin?"
)
print("No session:", request_no_session)

# test a full response with grounding
response = ChatResponse(
    answer="The recommended starting dose of metformin is 500mg twice daily.",
    grounding=[
        GroundingSource(
            source="who_diabetes_guidelines.pdf",
            quote="Metformin should be initiated at 500mg twice daily with meals.",
            score=0.21
        ),
        GroundingSource(
            source="drug_reference.pdf",
            quote="Maximum daily dose of metformin is 2550mg.",
            score=0.34
        )
    ]
)

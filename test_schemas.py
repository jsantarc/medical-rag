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
print("\nResponse:", response)
print("\nFirst source:", response.grounding[0].source)
print("Answer:", response.answer)

# test validation — this should fail
try:
    bad_request = ChatRequest(message="")
except Exception as e:
    print(f"\nValidation caught empty message: ){e}")
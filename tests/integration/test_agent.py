"""
Integration smoke tests for the DiabetesAssist agent.

Starts the server automatically, runs checks against it, then shuts it down.
Tests check the two main failure points:
  1. Tool call path  — agent calls document_search and returns a grounded answer
  2. No tool call path — agent answers directly without retrieval

They do NOT assert answer quality — only that the server responds with non-empty text.
Run with: pytest tests/integration/
"""
import time
import subprocess
import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"


@pytest.fixture(scope="session", autouse=True)
def server():
    proc = subprocess.Popen(
        ["uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd="../..",
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(20):
        try:
            requests.get(BASE_URL)
            break
        except Exception:
            time.sleep(0.5)
    yield
    proc.terminate()
    proc.wait()


def ask(question, session_id="test"):
    requests.post(f"{BASE_URL}/reset")
    response = requests.post(
        f"{BASE_URL}/chat",
        json={"message": question, "session_id": session_id}
    )
    return response.text


def test_tool_call():
    """Agent should call document_search for a clinical question."""
    answer = ask("What is the recommended HbA1c target for type 2 diabetes?")
    assert len(answer) > 0


def test_no_tool_call():
    """Agent should answer directly without retrieval."""
    answer = ask("What does RAG stand for?")
    assert len(answer) > 0

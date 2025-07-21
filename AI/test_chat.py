import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_chat_rag_flow():
    # Create a new session
    resp = client.post("/session-init")
    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    # Send a chat request
    chat_payload = {"session_id": session_id, "prompt": "What is the Schaeffler Group?"}
    resp = client.post("/chat", json=chat_payload)
    assert resp.status_code == 200
    # The response is streamed as JSON, so parse the last chunk
    # FastAPI TestClient will collect the full response
    data = resp.json()
    assert "answer" in data
    assert "context" in data
    assert isinstance(data["context"], list)
    assert len(data["answer"]) > 0 
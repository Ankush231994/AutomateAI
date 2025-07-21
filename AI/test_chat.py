import pytest
from fastapi.testclient import TestClient
from AI.app.main import app

client = TestClient(app)

def test_chat_rag_flow():
    resp = client.post("/session-init") # test client sends a POST request to the /session-init endpoint
    assert resp.status_code == 200 # test client receives a 200 status code
    session_id = resp.json()["session_id"] # test client receives a session_id

    # Send a chat request with the correct 'message' key
    chat_payload = {"session_id": session_id, "message": "What is the Schaeffler Group?"} # test client sends a POST request to the /chat endpoint
    resp = client.post("/chat", json=chat_payload)
    assert resp.status_code == 200
    
    # The response is streamed as JSON, so parse the last chunk
    # FastAPI TestClient will collect the full response
    data = resp.json()
    assert "answer" in data
    assert len(data["answer"]) > 0 
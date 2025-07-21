import pytest
from fastapi.testclient import TestClient
from AI.app.main import app

client = TestClient(app)


def test_delete_session_works_correctly():
    """
    Tests the full lifecycle of the /delete endpoint to ensure it works as expected.
    """
    
    resp_session = client.post("/session-init")
    assert resp_session.status_code == 200
    session_id_to_delete = resp_session.json()["session_id"]


    resp_delete = client.delete(f"/delete?session_id={session_id_to_delete}")


    assert resp_delete.status_code == 200
    assert resp_delete.json() == {"detail": "Session and messages deleted."}

    resp_history = client.get(f"/session-history?session_id={session_id_to_delete}")
    assert resp_history.status_code == 404 
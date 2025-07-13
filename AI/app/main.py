from fastapi import FastAPI, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlmodel import Session, select, delete
from AI.models import Conversation, Message
from AI.database import engine, create_db_and_tables
from langchain_community.llms import Ollama
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import re

app = FastAPI()

load_dotenv()
model_name = os.getenv("LLM_MODEL", "deepseek-r1:1.5b")
llm = Ollama(
    model=model_name,
    base_url="http://localhost:11434",
    timeout=60
)

class ChatRequest(BaseModel):
    session_id: int
    prompt: str

def extract_final_answer(text):
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

@app.on_event("startup")
def on_startup():
    create_db_and_tables()

@app.post("/session-init")
def session_init():
    with Session(engine) as session:
        convo = Conversation()
        session.add(convo)
        session.commit()
        session.refresh(convo)
        return {"session_id": convo.id}

@app.post("/chat")
def chat(request: ChatRequest):
    with Session(engine) as session:
        convo = session.get(Conversation, request.session_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Session not found.")
        # Fetch history for context
        messages = session.exec(
            select(Message).where(Message.session_id == request.session_id).order_by(Message.timestamp)
        ).all()
        history = "\n".join([
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.text}" for m in messages
        ])
        # Store user message
        user_msg = Message(session_id=request.session_id, role="user", text=request.prompt)
        session.add(user_msg)
        session.commit()
        session.refresh(user_msg)
        # Build prompt for LLM
        system_prompt = (
            "You are Automate AI, your AI assistant, here to assist you with your tasks. "
            "Never include any <think> tags, internal thoughts, or explanations of your reasoning. "
            "Respond ONLY with the final answer to the user's question, in a friendly, concise, and professional manner. "
            "Do not repeat the user's question. "
            "Do not output anything except your direct answer."
        )
        full_prompt = f"{system_prompt}\n{history}\nUser: {request.prompt}\nAssistant: "
        def llm_stream():
            response = ""
            for chunk in llm.stream(full_prompt):
                response += chunk
            cleaned_response = extract_final_answer(response)
            ai_msg = Message(session_id=request.session_id, role="assistant", text=cleaned_response)
            with Session(engine) as s2:
                s2.add(ai_msg)
                s2.commit()
            yield cleaned_response
        return StreamingResponse(llm_stream(), media_type="text/plain")

@app.get("/session-history")
def session_history(session_id: int = Query(...)):
    with Session(engine) as session:
        messages = session.exec(
            select(Message).where(Message.session_id == session_id).order_by(Message.timestamp)
        ).all()
        if not messages:
            raise HTTPException(status_code=404, detail="Session not found or has no messages.")
        return [
            {
                "id": m.id,
                "session_id": m.session_id,
                "role": m.role,
                "text": m.text,
                "timestamp": m.timestamp
            } for m in messages
        ]

@app.delete("/delete")
def delete_session(session_id: int = Query(...)):
    with Session(engine) as session:
        convo = session.get(Conversation, session_id)
        if not convo:
            raise HTTPException(status_code=404, detail="Session not found.")
        session.delete(convo)
        session.exec(delete(Message).where(Message.session_id == session_id))
        session.commit()
        return {"detail": "Session and messages deleted."} 

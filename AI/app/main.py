from fastapi import FastAPI, HTTPException, status, Query, APIRouter
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
import logging


app = FastAPI()

load_dotenv()
model_name = os.getenv("LLM_MODEL", "deepseek-r1:1.5b")
llm = Ollama(
    model=model_name,
    base_url="http://localhost:11434",
    timeout=60
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    session_id: int
    prompt: str

def extract_final_answer(text):
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()

@app.on_event("startup") #Used once, to run code at application startup
def on_startup():
    create_db_and_tables()

@router.post("/session-init")
def session_init():
    logger.info("Session started")
    try:
        with Session(engine) as session:
            convo = Conversation()
            session.add(convo)
            session.commit()
            session.refresh(convo)
            logger.info(f"Session created: session_id={convo.id}")
            return {"session_id": convo.id}
    except Exception as e:
        logger.error(f"Error in session_init: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during session initialization.")

@router.post("/chat")
def chat(request: ChatRequest):
    logger.info(f"Session started or continued: session_id={request.session_id}")
    logger.info(f"Chat request: session_id={request.session_id}, prompt={request.prompt}")
    try:
        with Session(engine) as session:
            convo = session.get(Conversation, request.session_id)
            if not convo:
                logger.error(f"Session not found: session_id={request.session_id}")
                raise HTTPException(status_code=404, detail="Session not found.")
            messages = session.exec(
                select(Message).where(Message.session_id == request.session_id).order_by(getattr(Message, "timestamp"))
            ).all()
            # Only include last 2 user/assistant pairs (4 messages)
            recent_messages = messages[-4:] if len(messages) >= 4 else messages
            history = "\n".join([
                f"{'User' if m.role == 'user' else 'Assistant'}: {m.text}" for m in recent_messages
            ])
            user_msg = Message(session_id=request.session_id, role="user", text=request.prompt)
            session.add(user_msg)
            session.commit()
            session.refresh(user_msg)
            system_prompt = (
                "You are Automate AI, your AI assistant, here to assist you with your tasks. "
                "Never include any <think> tags, internal thoughts, or explanations of your reasoning. "
                "Respond ONLY with the final answer to the user's question, in a friendly, concise, and professional manner. "
                "Do not repeat the user's question. "
                "Do not output anything except your direct answer."
            )
            full_prompt = f"{system_prompt}\n{history}\nUser: {request.prompt}\nAssistant: "
            logger.info(f"LLM prompt: {full_prompt}")
            def llm_stream():
                response = ""
                try:
                    for chunk in llm.stream(full_prompt):
                        response += chunk
                    cleaned_response = extract_final_answer(response)
                    logger.info(f"LLM response: {cleaned_response}")
                    ai_msg = Message(session_id=request.session_id, role="assistant", text=cleaned_response)
                    with Session(engine) as s2:
                        s2.add(ai_msg)
                        s2.commit()
                    yield cleaned_response
                except Exception as e:
                    logger.error(f"Error in LLM streaming: {e}")
                    yield "Internal server error during LLM response."
            return StreamingResponse(llm_stream(), media_type="text/plain")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during chat.")

@router.get("/session-history")
def session_history(session_id: int = Query(...)):
    logger.info(f"Session history requested: session_id={session_id}")
    try:
        with Session(engine) as session:
            messages = session.exec(
                select(Message).where(Message.session_id == session_id).order_by(getattr(Message, "timestamp"))
            ).all()
            if not messages:
                logger.warning(f"Session not found or has no messages: session_id={session_id}")
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in session_history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during session history retrieval.")

@router.delete("/delete")
def delete_session(session_id: int = Query(...)):
    logger.info(f"Delete session requested: session_id={session_id}")
    try:
        with Session(engine) as session:
            convo = session.get(Conversation, session_id)
            if not convo:
                logger.error(f"Session not found for deletion: session_id={session_id}")
                raise HTTPException(status_code=404, detail="Session not found.")
            session.delete(convo)
            session.execute(delete(Message).where(getattr(Message, "session_id") == session_id))
            session.commit()
            logger.info(f"Session and messages deleted: session_id={session_id}")
            return {"detail": "Session and messages deleted."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during session deletion.")

app.include_router(router) 

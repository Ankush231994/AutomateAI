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
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from AI.config import EMBEDDINGS_DIR, EMBEDDINGS_FILE, FAISS_INDEX_FILE, MODEL_NAME, OLLAMA_BASE_URL, OLLAMA_TIMEOUT, DB_URL


app = FastAPI()  ## FastAPI instance

load_dotenv()
model_name = os.getenv("LLM_MODEL", "deepseek-r1:1.5b") ## LLM model name
llm = Ollama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    timeout=OLLAMA_TIMEOUT
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()  ## API router used for routing the requests to the appropriate endpoints  

class ChatRequest(BaseModel):  # validates incoming chat requests
    session_id: int    # ensures session_id is an integer
    prompt: str        # ensures prompt is a string

# RAG setup
# Use imported config variables
with open(EMBEDDINGS_FILE, 'rb') as f:
    all_embeddings = pickle.load(f)
faiss_index = faiss.read_index(FAISS_INDEX_FILE)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_final_answer(text):  # extracts the final answer from the text and removes the <think> tags in the LLM response
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
            convo = Conversation()    #
            session.add(convo)   
            session.commit()
            session.refresh(convo)
            logger.info(f"Session created: session_id={convo.id}")
            return {"session_id": convo.id}
    except Exception as e:
        logger.error(f"Error in session_init: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during session initialization.")

@router.post("/chat")
def chat(request: ChatRequest):   # handles the chat requests
    logger.info(f"Session started or continued: session_id={request.session_id}")   
    logger.info(f"Chat request: session_id={request.session_id}, prompt={request.prompt}")
    try:
        with Session(engine) as session:
            convo = session.get(Conversation, request.session_id)  # gets the conversation from the database
            if not convo:
                logger.error(f"Session not found: session_id={request.session_id}") 
                raise HTTPException(status_code=404, detail="Session not found.")
            messages = session.exec( 
                select(Message).where(Message.session_id == request.session_id).order_by(getattr(Message, "timestamp"))  # gets the messages from the database
            ).all()
            # Only include last 2 user/assistant pairs (4 messages)
            recent_messages = messages[-4:] if len(messages) >= 4 else messages  # gets the last 4 messages
            history = "\n".join([  # joins the messages into a single string
                f"{'User' if m.role == 'user' else 'Assistant'}: {m.text}" for m in recent_messages  # formats the messages
            ])
            user_msg = Message(session_id=request.session_id, role="user", text=request.prompt)  # creates a new message
            session.add(user_msg) 
            session.commit() 
            session.refresh(user_msg)  
            # RAG context retrieval
            def retrieve_context(query, top_k=5): 
                query_emb = embed_model.encode(query).astype('float32').reshape(1, -1)
                D, I = faiss_index.search(query_emb, top_k)   # search the FAISS index for the query
                results = []
                for idx in I[0]:    
                    results.append({  # adds the results to the list
                        'filename': all_embeddings[idx]['filename'],  # adds the filename to the result
                        'text': all_embeddings[idx]['text']  # adds the text to the result
                    })
                return results
            rag_chunks = retrieve_context(request.prompt, top_k=5)  # retrieves the context
            rag_context = '\n\n'.join([c['text'] for c in rag_chunks])  # joins the context into a single string
            system_prompt = (  # system prompt for the LLM
                "You are Automate AI, your AI assistant, here to assist you with your tasks. "
                "Use only the provided context below to answer the user's question. "
                "If the answer is not in the context, say 'I don't know based on the provided information.' "
                "Never include any <think> tags, internal thoughts, or explanations of your reasoning. "
                "Respond ONLY with the final answer to the user's question, in a friendly, concise, and professional manner. "
                "Do not repeat the user's question. "
                "Do not output anything except your direct answer."
            )
            full_prompt = f"{system_prompt}\n\nContext:\n{rag_context}\n\n{history}\nUser: {request.prompt}\nAssistant: "
            logger.info(f"LLM prompt: {full_prompt[:1000]} ...")


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
                    # Yield both answer and context as a JSON string
                    import json
                    yield json.dumps({
                        "answer": cleaned_response,
                        "context": [c['text'] for c in rag_chunks]
                    })
                except Exception as e:
                    logger.error(f"Error in LLM streaming: {e}")
                    yield "Internal server error during LLM response."
            return StreamingResponse(llm_stream(), media_type="application/json")
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

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
from AI.config import EMBEDDINGS_DIR, EMBEDDINGS_FILE, FAISS_INDEX_FILE, MODEL_NAME, OLLAMA_BASE_URL, OLLAMA_TIMEOUT

# Load embeddings and FAISS index at startup
with open(EMBEDDINGS_FILE, 'rb') as f:
    all_embeddings = pickle.load(f)
index = faiss.read_index(FAISS_INDEX_FILE)
model = SentenceTransformer('all-MiniLM-L6-v2')
llm = Ollama(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    timeout=OLLAMA_TIMEOUT
)
system_prompt = (
    "You are Automate AI, an expert assistant. "
    "Use only the provided context to answer the user's question. "
    "If the answer is not in the context, say 'I don't know based on the provided information.' "
    "Be concise and professional."
)

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRequest(BaseModel):
    query: str
    top_k: int = 5

class RAGResponse(BaseModel):
    answer: str
    context: List[str]


def retrieve_context(query, top_k=5):
    query_emb = model.encode(query).astype('float32').reshape(1, -1)
    D, I = index.search(query_emb, top_k)
    results = []
    for idx in I[0]:
        results.append({
            'filename': all_embeddings[idx]['filename'],
            'text': all_embeddings[idx]['text']
        })
    return results

@app.post("/rag_chat", response_model=RAGResponse)
def rag_chat(request: RAGRequest):
    logger.info(f"Received RAG query: {request.query}")
    try:
        context_chunks = retrieve_context(request.query, top_k=request.top_k)
        context_text = '\n\n'.join([c['text'] for c in context_chunks])
        prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nUser: {request.query}\nAssistant: "
        answer = ""
        for chunk in llm.stream(prompt):
            answer += chunk
        logger.info("LLM answer generated.")
        return RAGResponse(
            answer=answer.strip(),
            context=[c['text'] for c in context_chunks]
        )
    except Exception as e:
        logger.error(f"Error in /rag_chat: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during RAG chat.") 
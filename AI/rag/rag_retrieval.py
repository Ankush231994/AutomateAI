import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama

EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), 'embeddings')
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, 'chunk_embeddings.pkl')
FAISS_INDEX_FILE = os.path.join(EMBEDDINGS_DIR, 'faiss.index')

# Load embeddings and FAISS index
with open(EMBEDDINGS_FILE, 'rb') as f:
    all_embeddings = pickle.load(f)
index = faiss.read_index(FAISS_INDEX_FILE)

model = SentenceTransformer('all-MiniLM-L6-v2')


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

if __name__ == "__main__":
    llm = Ollama(
        model="deepseek-r1:1.5b",
        base_url="http://localhost:11434",
        timeout=60
    )
    system_prompt = (
        "You are Automate AI, an expert assistant. "
        "Use only the provided context to answer the user's question. "
        "If the answer is not in the context, say 'I don't know based on the provided information.' "
        "Be concise and professional."
    )
    while True:
        query = input("Enter your query (or 'exit'): ").strip()
        if query.lower() == 'exit':
            break
        context_chunks = retrieve_context(query, top_k=5)
        print("\n--- Retrieved Context ---")
        for i, chunk in enumerate(context_chunks):
            print(f"[{i+1}] {chunk['filename']}")
            print(chunk['text'][:500], '...')
            print('-'*40)
        context_text = '\n\n'.join([c['text'] for c in context_chunks])
        prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nUser: {query}\nAssistant: "
        print("\n--- LLM Prompt (truncated) ---\n")
        print(prompt[:1500], '...')
        print("\n--- End of Prompt Preview ---\n")
        print("\n--- LLM Answer (streaming) ---\n")
        try:
            for chunk in llm.stream(prompt):
                print(chunk, end='', flush=True)
            print()  # Newline after answer
        except Exception as e:
            print(f"Error calling LLM: {e}")
        print("\n==============================\n") 
import os
import pickle
import numpy as np
import faiss
from AI.config import EMBEDDINGS_FILE, FAISS_INDEX_FILE

# Load embeddings
with open(EMBEDDINGS_FILE, 'rb') as f:
    all_embeddings = pickle.load(f)

# Prepare data for FAISS
embeddings = np.stack([np.array(item['embedding'], dtype=np.float32) for item in all_embeddings])

# Build FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save the index
faiss.write_index(index, FAISS_INDEX_FILE)
print(f"FAISS index saved to {FAISS_INDEX_FILE}")

# Demo search function
if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    while True:
        query = input("Enter your query (or 'exit'): ").strip()
        if query.lower() == 'exit':
            break
        query_emb = model.encode(query)
        if not isinstance(query_emb, np.ndarray):
            query_emb = np.array(query_emb)
        query_emb = query_emb.astype('float32').reshape(1, -1)
        D, I = index.search(query_emb, 5)  # Top 5
        print("Top results:")
        for rank, idx in enumerate(I[0]):
            print(f"[{rank+1}] {all_embeddings[idx]['filename']}")
            print(all_embeddings[idx]['text'][:300], '...')
            print('-'*40) 
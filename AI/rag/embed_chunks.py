import os
import pickle
from sentence_transformers import SentenceTransformer

CHUNKS_DIR = os.path.join(os.path.dirname(__file__), 'chunks')
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), 'embeddings')
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, 'chunk_embeddings.pkl')

model = SentenceTransformer('all-MiniLM-L6-v2')

all_embeddings = []

for fname in os.listdir(CHUNKS_DIR):
    if not fname.endswith('.txt'):
        continue
    with open(os.path.join(CHUNKS_DIR, fname), 'r', encoding='utf-8') as f:
        text = f.read().strip()
    if not text:
        continue
    embedding = model.encode(text)
    all_embeddings.append({
        'filename': fname,
        'text': text,
        'embedding': embedding
    })
    print(f"Embedded: {fname}")

with open(EMBEDDINGS_FILE, 'wb') as f:
    pickle.dump(all_embeddings, f)

print(f"All embeddings saved to {EMBEDDINGS_FILE}") 
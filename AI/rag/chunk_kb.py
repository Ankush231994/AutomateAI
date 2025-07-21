import os

# Use the KB directory at the project root
KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'KB'))
CHUNKS_DIR = os.path.join(os.path.dirname(__file__), 'chunks')
os.makedirs(CHUNKS_DIR, exist_ok=True)

CHUNK_SIZE = 1  # Number of paragraphs per chunk (set to 1 for fine granularity)


def chunk_text(text, chunk_size=CHUNK_SIZE):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks = []
    for i in range(0, len(paragraphs), chunk_size):
        chunk = '\n\n'.join(paragraphs[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def chunk_kb_files():
    for fname in os.listdir(KB_DIR):
        if not fname.endswith('.txt'):
            continue
        with open(os.path.join(KB_DIR, fname), 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            chunk_fname = f"{os.path.splitext(fname)[0]}_chunk{idx+1}.txt"
            with open(os.path.join(CHUNKS_DIR, chunk_fname), 'w', encoding='utf-8') as cf:
                cf.write(chunk)
        print(f"Chunked {fname} into {len(chunks)} chunks.")

if __name__ == "__main__":
    chunk_kb_files()
    print("Chunking complete. Chunks are saved in the 'chunks' directory.") 
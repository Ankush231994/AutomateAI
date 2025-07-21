import os
from AI.config import KB_DIR, CHUNKS_DIR

# Ensure the output directory exists
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
    print(f"Reading from KB_DIR: {KB_DIR}")
    print(f"Writing to CHUNKS_DIR: {CHUNKS_DIR}")
    for fname in os.listdir(KB_DIR):
        if not fname.endswith('.txt'):
            continue
        in_path = os.path.join(KB_DIR, fname)
        print(f"  Processing file: {in_path}")
        with open(in_path, 'r', encoding='utf-8') as f:
            text = f.read()
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            chunk_fname = f"{os.path.splitext(fname)[0]}_chunk{idx+1}.txt"
            out_path = os.path.join(CHUNKS_DIR, chunk_fname)
            print(f"    Writing chunk to: {out_path}")
            with open(out_path, 'w', encoding='utf-8') as cf:
                cf.write(chunk)
        print(f"Chunked {fname} into {len(chunks)} chunks.")

if __name__ == "__main__":
    chunk_kb_files()
    print("Chunking complete. Chunks are saved in the 'chunks' directory.") 
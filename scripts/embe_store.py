#!/usr/bin/env python3
"""
embed_store.py
Compute embeddings for chunks and build a FAISS index.

Usage:
python scripts/embed_store.py \
  --chunks_file chunks/chunks.jsonl \
  --index_dir index \
  --model_name all-MiniLM-L6-v2 \
  --batch_size 32
"""
import argparse, json, os
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def load_chunks(chunks_file: Path):
    rows = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def build_embeddings(rows, model, batch_size=32):
    texts = [r["text"] for r in rows]
    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]
        emb = model.encode(batch, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        all_embs.append(emb)
    return np.vstack(all_embs)

def build_faiss_index(embeddings, index_path):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product over normalized vectors ≈ cosine
    index.add(embeddings)
    faiss.write_index(index, str(index_path))
    return index

def save_meta(rows, meta_path):
    meta = [{"chunk_id": r["chunk_id"], "doc_id": r.get("doc_id"), "source_path": r.get("source_path")} for r in rows]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--chunks_file", required=True)
    p.add_argument("--index_dir", required=True)
    p.add_argument("--model_name", default="all-MiniLM-L6-v2")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    chunks_file = Path(args.chunks_file)
    idx_dir = Path(args.index_dir)
    idx_dir.mkdir(parents=True, exist_ok=True)

    rows = load_chunks(chunks_file)
    if not rows:
        raise SystemExit("No chunks found. Run Phase 1 first.")

    print("Loading model:", args.model_name)
    model = SentenceTransformer(args.model_name)

    print("Computing embeddings...")
    embeddings = build_embeddings(rows, model, batch_size=args.batch_size)
    print("Embeddings shape:", embeddings.shape)

    idx_path = idx_dir / "faiss_index.index"
    print("Building FAISS index...")
    build_faiss_index(embeddings, idx_path)
    print("Saved FAISS index to", idx_path)

    meta_path = idx_dir / "chunks_meta.json"
    save_meta(rows, meta_path)
    np.save(str(idx_dir / "embeddings.npy"), embeddings)
    print("Saved metadata and embeddings to", idx_dir)
    print("Done.")

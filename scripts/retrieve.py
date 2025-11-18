#!/usr/bin/env python3
"""
retrieve.py
Query the FAISS index and print top-k chunk previews.

Usage:
python scripts/retrieve.py --index_dir index --query "your question" --k 5
"""
import argparse, json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

def load_index(index_dir: Path):
    idx = faiss.read_index(str(index_dir / "faiss_index.index"))
    meta = json.loads((index_dir / "chunks_meta.json").read_text(encoding="utf-8"))
    return idx, meta

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index_dir", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--model_name", default="all-MiniLM-L6-v2")
    args = p.parse_args()

    idx_dir = Path(args.index_dir)
    index, meta = load_index(idx_dir)

    model = SentenceTransformer(args.model_name)
    q_emb = model.encode([args.query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, args.k)
    D = D[0].tolist(); I = I[0].tolist()

    print(f"\nTop {args.k} results for: {args.query}\n")
    for score, idx in zip(D, I):
        item = meta[idx]
        print(f"score={score:.4f}  chunk_id={item['chunk_id']}  doc={item.get('doc_id')}")
        # show preview (first 400 chars)
        with open("chunks/chunks.jsonl", "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i == idx:
                    rec = json.loads(line)
                    preview = rec["text"].replace("\n", " ")[:400]
                    print(preview)
                    break
    print()

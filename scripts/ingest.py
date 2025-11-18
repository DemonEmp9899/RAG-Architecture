#!/usr/bin/env python3
"""
ingest.py
Phase 1: Ingest documents (.txt, .pdf) from data_dir, extract text, clean, chunk into overlapping chunks,
and save chunks as JSONL + metadata CSV.

Usage:
python scripts/ingest.py --data_dir data --out_chunks chunks/chunks.jsonl --out_meta metadata/meta.csv --chunk_size 2000 --overlap 200
"""

import os
import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm
import pandas as pd

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\?\!])\s+')

def extract_text_from_pdf(path: Path) -> str:
    if fitz is None:
        raise RuntimeError("PyMuPDF not available. Install with: pip install PyMuPDF")
    doc = fitz.open(str(path))
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    return "\n".join(texts)

def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return extract_text_from_pdf(path)
    else:
        # assume text file
        return path.read_text(encoding="utf-8", errors="ignore")

def clean_text(text: str) -> str:
    # basic cleaning: normalize whitespaces, remove multiple newlines
    text = text.replace("\r", " ")
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = text.strip()
    return text

def sentence_split(text: str):
    # naive sentence split; for better quality use nltk.sent_tokenize
    sents = SENTENCE_SPLIT_RE.split(text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200):
    """
    chunk_size and overlap are in characters (simple). This keeps sentence boundaries.

    Returns list of dicts: [{"chunk": "...", "start_char": n, "end_char": m}, ...]
    """
    sentences = sentence_split(text)
    chunks = []
    cur = ""
    cur_start = 0
    char_pos = 0

    for sent in sentences:
        if not cur:
            cur_start = char_pos
        if len(cur) + len(sent) + 1 <= chunk_size:
            cur = (cur + " " + sent).strip()
        else:
            # finalize current chunk
            chunks.append({"chunk": cur, "start_char": cur_start, "end_char": char_pos})
            # start new chunk with overlap: keep last `overlap` chars from cur
            keep_chars = cur[-overlap:] if overlap > 0 else ""
            cur = (keep_chars + " " + sent).strip()
            cur_start = char_pos - len(keep_chars)
        char_pos += len(sent) + 1  # approximate char position
    if cur:
        chunks.append({"chunk": cur, "start_char": cur_start, "end_char": char_pos})
    return chunks

def ingest_folder(data_dir: Path, out_chunks: Path, out_meta: Path, chunk_size: int, overlap: int):
    out_chunks.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    total_chunks = 0

    with open(out_chunks, "w", encoding="utf-8") as fout:
        for file_path in tqdm(list(data_dir.glob("**/*"))):
            if file_path.is_dir():
                continue
            if file_path.suffix.lower() not in {".pdf", ".txt"}:
                continue
            try:
                text = extract_text(file_path)
            except Exception as e:
                print(f"Skipping {file_path} due to error: {e}")
                continue
            text = clean_text(text)
            if len(text) < 50:
                continue
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            doc_id = str(file_path.relative_to(data_dir))
            for i, c in enumerate(chunks):
                chunk_id = f"{doc_id}__chunk_{i:03d}"
                record = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "text": c["chunk"],
                    "start_char": c["start_char"],
                    "end_char": c["end_char"],
                    "source_path": str(file_path)
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                meta_rows.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "source_path": str(file_path),
                    "chunk_len_chars": len(c["chunk"])
                })
            total_chunks += len(chunks)

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(out_meta, index=False)
    print(f"Ingestion finished. Wrote {total_chunks} chunks to {out_chunks}. Metadata -> {out_meta}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_chunks", required=True)
    parser.add_argument("--out_meta", required=True)
    parser.add_argument("--chunk_size", type=int, default=2000)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    assert data_dir.exists(), "data_dir not found"

    ingest_folder(data_dir, Path(args.out_chunks), Path(args.out_meta), args.chunk_size, args.overlap)

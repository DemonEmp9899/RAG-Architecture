#!/usr/bin/env python3
"""
generate_answer.py

Modified to use a single env var:
  DEEPSEEK_API_KEY=<your-token>

Note:
- If your DEEPSEEK_API_KEY contains a GitHub PAT (github_pat_...), the script will
  attempt to use it as the bearer token for DeepSeek's HTTP API. This may work
  if the vendor accepts Marketplace tokens; otherwise you'll see a 401/403.
- On failure, the script falls back to a local Hugging Face model.

Usage examples:
# Use DeepSeek token from DEEPSEEK_API_KEY (auto mode will prefer DeepSeek if present)
python scripts/generate_answer.py --index_dir index --query "What is attention mechanism?" --provider auto

# Force local mode (no API calls)
python scripts/generate_answer.py --index_dir index --query "What is attention mechanism?" --provider local
"""
import argparse, json, os, textwrap
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from dotenv import load_dotenv
load_dotenv()

# optional: local hf
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
except Exception:
    pipeline = None

import requests

# ------------ Utilities (retrieval + prompt building) -------------
def load_index_and_meta(index_dir: Path):
    idx = faiss.read_index(str(index_dir / "faiss_index.index"))
    meta = json.loads((index_dir / "chunks_meta.json").read_text(encoding="utf-8"))
    return idx, meta

def load_chunk_text_by_index(idx_num):
    with open("chunks/chunks.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx_num:
                return json.loads(line)["text"]
    return ""

def retrieve_top_k(query: str, model, index, k=4):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, k)
    return D[0].tolist(), I[0].tolist()

def build_prompt_basic(question: str, meta, indices, scores, include_preview_chars=600):
    ctx_parts = []
    citations = []
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        m = meta[idx]
        text = load_chunk_text_by_index(idx).replace("\n"," ")
        preview = text[:include_preview_chars]
        ctx_parts.append(f"--- Context [{rank}] (score={score:.3f}) Source: {m.get('doc_id')}\n{preview}")
        citations.append(f"[{rank}] {m.get('doc_id','unknown')}:{m.get('chunk_id')}")
    context_block = "\n\n".join(ctx_parts)
    citation_list = ", ".join(citations)
    prompt = textwrap.dedent(f"""
    You are a helpful assistant that answers questions using ONLY the provided context snippets.
    If the answer is not present in the snippets, say "I don't know" and do NOT hallucinate.
    Provide the answer concisely and include inline citations (e.g., [1], [2]) when using context.

    CONTEXT SNIPPETS:
    {context_block}

    QUESTION:
    {question}

    INSTRUCTIONS:
    1) Answer using ONLY the above context.
    2) After the answer, list the citations you used in the format: "Citations used: [1], [2]".
    3) If answer not found, reply: "I don't know based on the provided sources."
    """).strip()
    return prompt, citation_list

# ------------ OpenAI call (kept for completeness; uses OPENAI_API_KEY if present) -------------
def call_openai_chat(prompt: str, model_name="gpt-3.5-turbo", max_tokens=256, temperature=0.0):
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package missing. Install with: python -m pip install --upgrade openai") from e
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")
    client = OpenAI(api_key=api_key)
    messages = [
        {"role":"system","content":"You are a helpful assistant that answers using only the provided context."},
        {"role":"user","content":prompt}
    ]
    resp = client.chat.completions.create(model=model_name, messages=messages, max_tokens=max_tokens, temperature=temperature)
    choice0 = resp.choices[0]
    msg = getattr(choice0, "message", None)
    if msg is None:
        assistant = choice0.get("message", {}).get("content") if isinstance(choice0, dict) else str(choice0)
    else:
        assistant = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", str(choice0))
    return (assistant or "").strip()

# ------------ DeepSeek using single DEEPSEEK_API_KEY env var -------------
def call_deepseek_with_key(prompt: str, model="deepseek-chat", max_tokens=256, temperature=0.0):
    """
    Uses the DEEPSEEK_API_KEY env var as the Bearer token.
    DEEPSEEK_API_KEY may contain a GitHub PAT (github_pat_...) or a vendor-issued deepseek- key.
    """
    token = os.getenv("DEEPSEEK_API_KEY")
    if not token:
        raise RuntimeError("DEEPSEEK_API_KEY not set in environment (.env). Put your token there.")
    # Example DeepSeek endpoint — update if vendor provided a different URL
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content":"You are a helpful assistant that answers using only the provided context."},
            {"role":"user","content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if resp.status_code == 200:
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            # fallback: return a JSON snippet if structure differs
            return json.dumps(data)[:2000]
    else:
        # Provide detailed error for debugging, then raise
        raise RuntimeError(f"DeepSeek API returned {resp.status_code}: {resp.text}")

# ------------ Local HF fallback -------------
def call_local_model(prompt: str, hf_model_name="google/flan-t5-small", max_new_tokens=256):
    if pipeline is None:
        raise RuntimeError("transformers not installed. Run: pip install transformers accelerate torch")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    # try seq2seq
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
        gen = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
        out = gen(prompt, max_length=max_new_tokens, do_sample=False)
        return out[0]["generated_text"].strip()
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(hf_model_name)
        gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
        out = gen(prompt, max_length=max_new_tokens, do_sample=False)
        return out[0]["generated_text"].strip()

# ------------ Main -------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index_dir", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--provider", choices=["openai","deepseek","local","auto"], default="auto",
                   help="Which provider to use. 'auto' prefers DEEPSEEK_API_KEY -> OPENAI_API_KEY -> local")
    p.add_argument("--openai_model", default="gpt-3.5-turbo")
    p.add_argument("--deepseek_model", default="deepseek-chat")
    p.add_argument("--hf_model_name", default="google/flan-t5-small")
    p.add_argument("--max_tokens", type=int, default=256)
    args = p.parse_args()

    index_dir = Path(args.index_dir)
    index, meta = load_index_and_meta(index_dir)

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    scores, indices = retrieve_top_k(args.query, embedder, index, k=args.k)

    prompt, citation_list = build_prompt_basic(args.query, meta, indices, scores, include_preview_chars=600)

    print("\n--- Retrieved Sources ---")
    for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
        print(f"[{rank}] doc={meta[idx].get('doc_id')} chunk={meta[idx].get('chunk_id')} score={score:.4f}")

    print("\n--- Prompt Preview (truncated) ---")
    print(prompt[:1200] + ("\n... (truncated) ..." if len(prompt) > 1200 else ""))

    # decide provider
    prov = args.provider
    if prov == "auto":
        if os.getenv("DEEPSEEK_API_KEY"):
            prov = "deepseek"
        elif os.getenv("OPENAI_API_KEY"):
            prov = "openai"
        else:
            prov = "local"

    ans = None
    try:
        if prov == "openai":
            ans = call_openai_chat(prompt, model_name=args.openai_model, max_tokens=args.max_tokens)
        elif prov == "deepseek":
            try:
                ans = call_deepseek_with_key(prompt, model=args.deepseek_model, max_tokens=args.max_tokens)
            except Exception as e:
                print("DeepSeek call failed:", e)
                print("If the token is a GitHub PAT and the vendor does not accept it directly, you'll see 401/403.")
                print("Falling back to local model...")
                ans = call_local_model(prompt, hf_model_name=args.hf_model_name, max_new_tokens=args.max_tokens)
        else:
            ans = call_local_model(prompt, hf_model_name=args.hf_model_name, max_new_tokens=args.max_tokens)
    except Exception as e:
        print("Provider call failed:", e)
        print("Falling back to local model...")
        ans = call_local_model(prompt, hf_model_name=args.hf_model_name, max_new_tokens=args.max_tokens)

    print("\n--- Answer ---")
    print(ans)
    print("\n--- Citations Suggested by Retrieval ---")
    print(citation_list)
    print("\n--- End ---\n")

if __name__ == "__main__":
    main()

✨ RAG-Based AI Knowledge System
🔍 Your documents. Your knowledge. Your AI — fully powered by Retrieval-Augmented Generation.

Welcome to the RAG-powered AI pipeline that blends semantic search, embeddings, vector databases, and LLMs to give you an intelligent, context-aware answering system that never hallucinates and always stays grounded in your data.

🚀 What This Project Does

This system builds a complete RAG workflow:

🧩 Document ingestion & chunking
🧠 Embedding generation using OpenAI / SentenceTransformers
📦 Vector storage using FAISS / Chroma / Pinecone
🔎 Semantic retrieval (Top-k search)
🤖 LLM-powered generation (GPT / LLaMA / Local models)
🌐 Optional API server for production use

Basically — drag in your PDFs, and your AI becomes an expert on them.

🌈 Why This is Awesome

✨ No hallucinations — answers always come from your documents
🔄 Search + AI = smart memory system
⚡ Fast and lightweight — ready for production
📚 Scales beautifully with more documents
🛠️ 100% customizable — change vector DB, LLM, chunk sizes, etc.

🧠 Architecture (Simple + Beautiful)
📄 Documents → 🔪 Chunking → 🧠 Embeddings → 📦 Vector DB   → 🔍 Retriever → 🤖 LLM → 💬 Final Answer


The classic RAG pipeline — but cleaner, smarter, and yours.

🛠️ Setup Guide
🔧 STEP 1 — Install dependencies
pip install -r requirements.txt

🧾 STEP 2 — Create .env

Your keys & configs go here:

OPENAI_API_KEY=your_key
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o
VECTOR_STORE=faiss
CHUNK_SIZE=500
CHUNK_OVERLAP=50

📥 STEP 3 — Add Your Documents

Drop files into:

data/


Supports: PDF, TXT, MD, DOCX

🧱 STEP 4 — Ingest Documents
python scripts/ingest.py

🤖 STEP 5 — Run the RAG Server
python scripts/embe_store.py
python scripts/retrieve.py

Open Swagger UI:
👉 http://localhost:8000/docs

🔍 STEP 6 — Query the System
python scripts/generate_answer.py --question "Explain CNN vs RNN"

🎉 You Now Have a Smart AI That Reads Your Documents!

Go crazy, build chatbots, internal search engines, research assistants — whatever you want.

👤 Author
Rudra Pratap Tomer
📧 rudratomer3@gmail.com

⭐ Love the project? Give it a star!

✨ “RAG turns your documents into intelligence. This repo turns RAG into reality.”


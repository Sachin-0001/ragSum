# RAG Chatbot + Text Summarizer

A simple multipage Streamlit app that includes:

- **RAG Chatbot** — Chat with your uploaded PDF files (upload → embed → retrieve → answer).
- **Text Summarizer** — Summarize long text using a fine-tuned T5 model.

Built with Streamlit, LangChain, FAISS, HuggingFace Transformers and optional Groq LLM integration.

---

## Features

- Upload PDFs and ask questions using a RAG pipeline (chunking + embeddings + FAISS).
- Chat-style interface with message history.
- Copy-to-clipboard icon for assistant answers.
- Separate Summarizer page (T5-based).
- Multipage app using Streamlit `pages/` folder — deploys easily to Streamlit Cloud.

---
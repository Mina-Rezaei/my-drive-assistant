# 🤖 My Drive Assistant

A personal RAG-based AI assistant that answers questions from my personal Google Drive documents.

## Tech Stack
- Google Drive API — document ingestion
- sentence-transformers — embeddings
- ChromaDB — vector store
- HuggingFace Inference API — LLM
- Gradio — chat interface
- HuggingFace Spaces — hosting

## How it works
1. Connects to my Google Drive
2. Extracts text from my documents
3. Chunks and embeds the content
4. Uses semantic search to find relevant chunks
5. Answers questions using an LLM grounded in my documents

## Setup
1. Clone the repo
2. Add my `credentials.json` from Google Cloud Console
3. Add my HuggingFace token to `.env`
4. Run `python app.py`

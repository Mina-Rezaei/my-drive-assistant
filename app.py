import os
import gradio as gr
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from sentence_transformers import SentenceTransformer
import chromadb
from huggingface_hub import InferenceClient
import io
import pickle

# Google Drive scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Initialise models
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("my_drive_docs")
llm_client = InferenceClient(
    provider="auto",
    api_key=os.environ.get("HUGGINGFACE_TOKEN")
)

def authenticate_google_drive():
    """Authenticate with Google Drive"""
    creds = None
    if os.path.exists('token.json'):
        with open('token.json', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'wb') as token:
            pickle.dump(creds, token)
    return build('drive', 'v3', credentials=creds)

def fetch_drive_documents(service):
    """Fetch text documents from Google Drive"""
    results = service.files().list(
        q="mimeType='text/plain' or mimeType='application/vnd.google-apps.document'",
        pageSize=50,
        fields="files(id, name, mimeType)"
    ).execute()
    return results.get('files', [])

def download_file(service, file_id, mime_type):
    """Download file content"""
    try:
        if mime_type == 'application/vnd.google-apps.document':
            request = service.files().export_media(
                fileId=file_id,
                mimeType='text/plain'
            )
        else:
            request = service.files().get_media(fileId=file_id)
        
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return fh.getvalue().decode('utf-8', errors='ignore')
    except:
        return ""

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into chunks"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks

def index_documents():
    """Fetch, chunk and index all Drive documents"""
    service = authenticate_google_drive()
    files = fetch_drive_documents(service)
    
    total_chunks = 0
    for file in files:
        content = download_file(service, file['id'], file['mimeType'])
        if not content.strip():
            continue
        
        chunks = chunk_text(content)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file['id']}_{i}"
            embedding = embedding_model.encode(chunk).tolist()
            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{"filename": file['name'], "chunk_order": i}]
            )
            total_chunks += 1
    
    return f"✅ Indexed {len(files)} documents — {total_chunks} chunks"

def ask_question(question):
    """RAG — retrieve and answer"""
    if collection.count() == 0:
        return "Please index your documents first by clicking 'Index My Drive'."
    
    # Retrieve relevant chunks
    query_embedding = embedding_model.encode(question).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )
    
    chunks = results['documents'][0]
    filenames = [m['filename'] for m in results['metadatas'][0]]
    context = "\n\n".join(chunks)
    
    # Generate answer
    messages = [
        {
            "role": "system",
            "content": "You are a personal assistant with access to the user's documents. Answer based ONLY on the context provided. If the answer is not in the context, say 'I don't have that information in your documents.'"
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}"
        }
    ]
    
    response = llm_client.chat.completions.create(
        model="Qwen/Qwen3-235B-A22B",
        messages=messages,
        max_tokens=300
    )
    
    content = response.choices[0].message.content
    if not content:
        reasoning = getattr(response.choices[0].message, 'reasoning_content', None)
        content = reasoning or "No response generated"
    
    sources = list(set(filenames))
    return f"{content.strip()}\n\n📄 Sources: {', '.join(sources)}"

# Gradio UI
with gr.Blocks(title="My Drive Assistant") as demo:
    gr.Markdown("# 🤖 My Drive Assistant")
    gr.Markdown("Ask me anything about your Google Drive documents!")
    
    with gr.Row():
        index_btn = gr.Button("📁 Index My Drive", variant="primary")
        index_status = gr.Textbox(label="Status", interactive=False)
    
    chatbot = gr.Chatbot(height=400)
    question_input = gr.Textbox(
        placeholder="Ask a question about your documents...",
        label="Your question"
    )
    ask_btn = gr.Button("Ask", variant="primary")
    
    def chat(question, history):
        answer = ask_question(question)
        history.append((question, answer))
        return "", history
    
    index_btn.click(index_documents, outputs=index_status)
    ask_btn.click(chat, inputs=[question_input, chatbot], outputs=[question_input, chatbot])
    question_input.submit(chat, inputs=[question_input, chatbot], outputs=[question_input, chatbot])

demo.launch()

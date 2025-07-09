# import libraries
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from typing import List, Optional
from pydantic import BaseModel
import os
import uuid
import json
from dotenv import load_dotenv
from docx import Document
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# Load env variables
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_API_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME")
FAISS_INDEX_PATH = "faiss_index"
CHAT_DIR = "chats"

os.makedirs(CHAT_DIR, exist_ok=True)

app = FastAPI()

def get_text_from_file(file):
    if file.filename.endswith(".txt"):
        return file.file.read().decode("utf-8")
    elif file.filename.endswith(".docx"):
        doc = Document(file.file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file.file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        text = text.replace("-\n", "").replace("\n", " ").replace("  ", " ")
        return text
    return ""

def get_text_chunks(text_chunks):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    return splitter.split_text("\n".join(text_chunks))

def get_vector_store(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)

def get_chat_path(user_id, conversation_id):
    return os.path.join(CHAT_DIR, f"{user_id}_{conversation_id}.json")

def load_chat_history(user_id, conversation_id):
    path = get_chat_path(user_id, conversation_id)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return []

def save_chat_message(user_id, conversation_id, role, content):
    path = get_chat_path(user_id, conversation_id)
    chat = load_chat_history(user_id, conversation_id)
    chat.append({"role": role, "content": content})
    with open(path, "w") as f:
        json.dump(chat, f)

def build_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant tasked with answering questions based only on the provided context.
        If the context clearly does not contain the answer, say: \"Sorry, I do not know.\"
        But if the context contains scattered references or hints, you should infer the best answer using reasoning.
        If asked whether something is complete or if something is missing, explain based on what the context includes — and whether it looks exhaustive or partial.
        Try to provide 3–4 follow-up questions if the context supports it. If not, you may include fewer.
        Do not fabricate any information.

        Respond in the following format:
        Answer: <your answer here>

        Follow-up Questions:
        1. ...
        2. ...
        3. ...

        Context: {context}"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_API_DEPLOYMENT_NAME,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.0
    )

    return prompt | llm

chain = build_chain()

class ChatRequest(BaseModel):
    question: str
    user_id: Optional[str] = "user_" + str(uuid.uuid4())[:8]
    conversation_id: Optional[str] = "session_" + str(uuid.uuid4())[:8]

@app.post("/upload/", tags=["Upload file"])
async def upload_files(files: List[UploadFile] = File(...)):
    raw_text = []
    for file in files:
        raw_text.append(get_text_from_file(file))

    if raw_text:
        chunks = get_text_chunks(raw_text)
        get_vector_store(chunks)
        return {"message": "✅ Documents processed and vector store created."}
    else:
        return {"message": "⚠️ No valid files uploaded."}

@app.post("/ask/", tags=["Ask questions"])
async def ask_question(request: ChatRequest):
    if not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        return {"error": "❌ FAISS index not found. Please upload and process documents first."}

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(request.question, k=6)
    context = "\n\n".join([doc.page_content for doc in docs])

    history = load_chat_history(request.user_id, request.conversation_id)

    response = chain.invoke(
        {"question": request.question, "context": context, "chat_history": history}
    )

    save_chat_message(request.user_id, request.conversation_id, "user", request.question)
    save_chat_message(request.user_id, request.conversation_id, "assistant", response.content)

    return {
        "answer": response.content,
        "user_id": request.user_id,
        "conversation_id": request.conversation_id
    }

@app.get("/download/{user_id}/{conversation_id}", tags=["Download chat pdf"])
async def download_chat(user_id: str, conversation_id: str):
    messages = load_chat_history(user_id, conversation_id)
    if not messages:
        return {"error": "No chat history found for the given user and conversation."}

    filename = f"{user_id}_{conversation_id}.pdf"
    path = os.path.join(CHAT_DIR, filename)

    pdf = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    y = height - 40

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, "Conversation Transcript")
    y -= 20
    pdf.setFont("Helvetica", 12)
    pdf.drawString(40, y, f"User ID: {user_id}")
    y -= 15
    pdf.drawString(40, y, f"Conversation ID: {conversation_id}")
    y -= 25

    for msg in messages:
        if y < 60:
            pdf.showPage()
            y = height - 40
        role = msg["role"]
        content = msg["content"]
        icon = "👤" if role == "user" else "🤖"

        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(40, y, f"{icon} {role.capitalize()}:")
        y -= 15
        pdf.setFont("Helvetica", 11)
        for line in content.split("\n"):
            while line:
                chunk = line[:90]
                pdf.drawString(60, y, chunk)
                line = line[90:]
                y -= 14
            y -= 6

    pdf.save()
    return FileResponse(path, media_type="application/pdf", filename=filename)
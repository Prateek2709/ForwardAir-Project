# import necessary libraries
import os
from dotenv import load_dotenv
from docx import Document
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import AzureChatOpenAI
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import ConfigurableFieldSpec
import streamlit as st
import uuid
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# Load environment variables
load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
AZURE_OPENAI_API_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME")
FAISS_INDEX_PATH = "faiss_index"

# define the LLM
llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        azure_deployment=AZURE_OPENAI_API_DEPLOYMENT_NAME,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0.0
)

# Function to extract text from TXT, DOCX and PDF files
def get_text_from_file(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".docx"):
        doc = Document(file)
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        text = text.replace("-\n", "").replace("\n", " ").replace("  ", " ")
        return text
    return

# Function to split text into smaller chunks
def get_text_chunks(text_chunks):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_text("\n".join(text_chunks))

# Function to create FAISS vector store
def get_vector_store(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(FAISS_INDEX_PATH)
    st.success("Knowledge base created! You can now process your questions.")

# Session-based memory store
store = {}

def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = InMemoryChatMessageHistory()
    return store[(user_id, conversation_id)]

# Define the RAG chain with memory
def build_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant tasked with answering questions based only on the provided context.
                    If the context clearly does not contain the answer, say: \"Sorry, I do not know.\"
                    Strictly do not fabricate any kind of information.
                    But if the question asked is such that the answer is present in the context, but in a scattered way and not direct references, search it properly and then answer, instead of saying \"Sorry, I do not know.\"
                    If asked whether something is complete or if something is missing, explain only based on what the context includes — and whether it looks exhaustive or partial.
                    Once again, strictly refuse to answer any question that is not based on the context provided and answer should only be based on the context.
                    Try to provide 2-3 follow-up questions that are relevant to both the previous question asked and the context.

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

    return RunnableWithMessageHistory(
        prompt | llm,
        get_session_history=get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        history_factory_config=[
            ConfigurableFieldSpec(
                id="user_id",
                annotation=str,
                name="User ID",
                description="Unique identifier for the user.",
                default="user1",
                is_shared=True,
            ),
            ConfigurableFieldSpec(
                id="conversation_id",
                annotation=str,
                name="Conversation ID",
                description="Unique identifier for the conversation.",
                default="session1",
                is_shared=True,
            ),
        ]
    )

# Generate PDF from conversation
def generate_chat_pdf(user_id, conversation_id, messages):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 40

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(40, y, f"Conversation Transcript")
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
        icon = "\U0001F464" if role == "user" else "\U0001F916"

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
    buffer.seek(0)
    return buffer

# Streamlit App
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

def main():
    st.set_page_config(page_title="Chat with your documents")
    st.title("\U0001F4C4\U0001F4AC ForwardAir Chat Assistant")

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user_{str(uuid.uuid4())[:8]}"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chain" not in st.session_state:
        st.session_state.chain = build_chain()

    with st.sidebar:
        st.header("Upload & Process Files")
        docs = st.file_uploader("Upload your TXT/DOCX/PDF here \U0001F447", type=["txt", "docx", "pdf"], accept_multiple_files=True)

        if st.button("Build Knowledge Base"):
            raw_text = []
            if docs:
                for file in docs:
                    raw_text.append(get_text_from_file(file))
                if raw_text:
                    chunks = get_text_chunks(raw_text)
                    get_vector_store(chunks)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about your document")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        if not os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
            st.warning("Please build the knowledge base first.")
            return

        db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        context_docs = db.similarity_search(user_input, k=3)
        context = "\n\n".join([doc.page_content for doc in context_docs])

        response = st.session_state.chain.invoke(
            {"question": user_input, "context": context},
            config={"configurable": {
                "user_id": st.session_state.user_id,
                "conversation_id": st.session_state.session_id
            }}
        )

        st.session_state.messages.append({"role": "assistant", "content": response.content})
        with st.chat_message("assistant"):
            st.markdown(response.content)

    # Chat download button
    if st.session_state.messages:
        pdf_buffer = generate_chat_pdf(
            st.session_state.user_id,
            st.session_state.session_id,
            st.session_state.messages
        )
        st.download_button(
            label="\U0001F4C4 Download conversation file",
            data=pdf_buffer,
            file_name=f"{st.session_state.user_id}_{st.session_state.session_id[:8]}.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
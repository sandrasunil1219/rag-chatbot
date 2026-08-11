"""
RAG Chatbot - upload PDFs and images, ask questions, get grounded answers.

Stack:
    - Vector DB: Chroma (local)
    - Embeddings: sentence-transformers (local, free, no API key)
    - LLM: Groq (free API, OpenAI-compatible, needs GROQ_API_KEY)

Setup:
    pip install streamlit langchain langchain-community langchain-groq \
        langchain-huggingface langchain-text-splitters langchain-classic \
        chromadb pypdf pytesseract pillow sentence-transformers python-dotenv

    # System dependency for image OCR:
    #   Windows: install Tesseract-OCR (see tesseract_cmd path below)
    #   Ubuntu/Debian: sudo apt-get install tesseract-ocr
    #   Mac:           brew install tesseract

    # Set your free Groq API key (https://console.groq.com/keys):
    #   Create a file named ".env" in this same folder containing:
    #       GROQ_API_KEY=gsk_your_actual_key_here
    #   (Alternatively set it as a real environment variable instead of using .env)

    streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import pytesseract

# --- Load GROQ_API_KEY (and any other secrets) from a local .env file, if present ---
load_dotenv()

# --- Windows only: point pytesseract at the installed Tesseract binary ---
if os.name == "nt":
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.documents import Document

st.set_page_config(page_title="Document Q&A Chatbot", layout="wide")
st.title("📄 Ask Your Documents")

# ---------- Fail fast with a clear message if the API key is missing ----------
if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY is not set. Create a `.env` file in this folder with:\n\n"
        "```\nGROQ_API_KEY=gsk_your_actual_key_here\n```\n\n"
        "Get a free key at https://console.groq.com/keys, then restart the app."
    )
    st.stop()

# ---------- Session state ----------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------- Ingestion helpers ----------

def load_pdf(file_path: str) -> list[Document]:
    """Extract text per page from a PDF."""
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_image(file_path: str, filename: str) -> list[Document]:
    """OCR an image into a single text Document."""
    text = pytesseract.image_to_string(Image.open(file_path))
    if not text.strip():
        text = "(No readable text found in this image.)"
    return [Document(page_content=text, metadata={"source": filename})]


@st.cache_resource
def get_embeddings():
    # Runs locally on your machine, no API key or internet call needed after first download.
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_vectorstore(all_docs: list[Document]) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(all_docs)
    embeddings = get_embeddings()
    return Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")


# ---------- Sidebar: upload & index ----------
with st.sidebar:
    st.header("Upload documents")
    files = st.file_uploader(
        "PDFs or images",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if st.button("Build knowledge base") and files:
        with st.spinner("Reading and indexing documents..."):
            all_docs: list[Document] = []
            for f in files:
                suffix = os.path.splitext(f.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name

                if suffix.lower() == ".pdf":
                    all_docs.extend(load_pdf(tmp_path))
                else:
                    all_docs.extend(load_image(tmp_path, f.name))

                os.unlink(tmp_path)

            st.session_state.vectorstore = build_vectorstore(all_docs)
        st.success(f"Indexed {len(files)} file(s). Ask away!")

# ---------- Main: chat ----------
question = st.chat_input("Ask a question about your uploaded documents...")

if question:
    if st.session_state.vectorstore is None:
        st.warning("Upload and index a document first.")
    else:
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
        )
        with st.spinner("Thinking..."):
            result = qa_chain.invoke({"query": question})

        st.session_state.chat_history.append((question, result["result"], result["source_documents"]))

for q, a, sources in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)
        with st.expander("Sources"):
            for doc in sources:
                st.caption(f"{doc.metadata.get('source', 'document')} — {doc.page_content[:200]}...")
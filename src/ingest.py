# src/ingest.py
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# --- Config ---
PDF_DIR = "../data/pdfs"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "text-embedding-3-small"  # change per provider
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def load_pdfs(pdf_dir=PDF_DIR):
    docs = []
    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, fname)
        loader = PyPDFLoader(path)
        pages = loader.load()
        for i, doc in enumerate(pages):
            # attach metadata so we can show source later
            doc.metadata["source"] = fname
            doc.metadata["page"] = i+1
            docs.append(doc)
    return docs

def split_docs(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split = []
    for d in docs:
        pieces = splitter.split_documents([d])
        split.extend(pieces)
    return split

def create_chroma_collection(pieces, persist_directory=CHROMA_DIR):
    # start chroma client (local persist)
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))

    # Use OpenAIEmbeddings via embedding_functions wrapper or LangChain adapter
    ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)

    collection = client.create_collection(name="human_services", embedding_function=ef)
    # prepare metadata and docs
    metadatas = [p.metadata for p in pieces]
    texts = [p.page_content for p in pieces]
    ids = [f"doc_{i}" for i in range(len(texts))]
    collection.add(documents=texts, metadatas=metadatas, ids=ids)
    client.persist()
    return collection

if __name__ == "__main__":
    docs = load_pdfs()
    print(f"Loaded {len(docs)} pages from PDFs.")
    pieces = split_docs(docs, chunk_size=800, chunk_overlap=150)
    print(f"Split into {len(pieces)} chunks.")
    create_chroma_collection(pieces)
    print("Ingestion done.")

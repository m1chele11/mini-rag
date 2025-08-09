# src/ingest.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
import chromadb
from chromadb import PersistentClient
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv



load_dotenv() 

# --- Config ---
PDF_DIR = "../data/pdfs"
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMENSIONS = 768  
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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

def get_gemini_embeddings(texts, batch_size=100):
    """Generate embeddings using Gemini in batches"""
    embeddings = []
    
    # Process in batches to avoid API limits
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
        batch = texts[i:i + batch_size]
        
        try:
            # Use the correct API for google-generativeai
            batch_embeddings = []
            for text in batch:
                result = genai.embed_content(
                    model=EMBED_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                
                # Get embedding values and normalize
                embedding_values = np.array(result['embedding'])
                # Normalize the embedding (important for semantic similarity)
                normalized_embedding = embedding_values / np.linalg.norm(embedding_values)
                batch_embeddings.append(normalized_embedding.tolist())
            
            embeddings.extend(batch_embeddings)
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size + 1}: {e}")
            # Add zero embeddings for failed batch to maintain alignment
            batch_embeddings = [[0.0] * 768] * len(batch)
            embeddings.extend(batch_embeddings)
    
    return embeddings

def create_chroma_collection(pieces, persist_directory=CHROMA_DIR):

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    
    # Configure Gemini
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Use the PersistentClient for ChromaDB
    chroma_client = PersistentClient(path=persist_directory)

    try:
        collection = chroma_client.get_collection(name="human_services")
        chroma_client.delete_collection(name="human_services")  
        print("Deleted existing collection")
    except:
        pass  # Collection doesn't exist, which is fine
    
    # Create collection without embedding function (we'll handle embeddings manually)
    collection = chroma_client.create_collection(name="human_services")
    
    # prepare metadata and docs
    metadatas = [p.metadata for p in pieces]
    texts = [p.page_content for p in pieces]
    ids = [f"doc_{i}" for i in range(len(texts))]
    
    print(f"Generating embeddings for {len(texts)} documents using Gemini...")
    # Generate embeddings using Gemini
    text_embeddings = get_gemini_embeddings(texts, batch_size=50)
    
    print("Adding documents to ChromaDB...")
    collection.add(
        documents=texts, 
        metadatas=metadatas, 
        ids=ids,
        embeddings=text_embeddings
    )
    
    print(f"Successfully added {len(texts)} documents to collection")
    return collection

if __name__ == "__main__":
    docs = load_pdfs()
    print(f"Loaded {len(docs)} pages from PDFs.")
    pieces = split_docs(docs, chunk_size=800, chunk_overlap=150)
    print(f"Split into {len(pieces)} chunks.")
    create_chroma_collection(pieces)
    print("Ingestion done.")
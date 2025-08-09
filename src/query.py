# src/query.py
import os
from dotenv import load_dotenv
import google.generativeai as genai
import chromadb
from chromadb import PersistentClient
from openai import OpenAI
import numpy as np

load_dotenv()

OPENAI_API_KEY = os.getenv("APIKEY")  # For OpenRouter
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # For Gemini embeddings
CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMENSIONS = 768

# OpenRouter client for LLM responses
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
)

# Gemini configuration
genai.configure(api_key=GOOGLE_API_KEY)

def get_query_embedding(query_text):
    """Generate embedding for a query using Gemini"""
    try:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=query_text,
            task_type="retrieval_query"
        )
        
        # Get the embedding and normalize it
        embedding_values = np.array(result['embedding'])
        normalized_embedding = embedding_values / np.linalg.norm(embedding_values)
        return normalized_embedding.tolist()
        
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return None

def get_top_k(query, k=4):
    """Retrieve top-k similar documents using Gemini embeddings"""
    # Get query embedding
    query_embedding = get_query_embedding(query)
    if query_embedding is None:
        return []
    
    # Query ChromaDB with the embedding
    client = PersistentClient(path=CHROMA_DIR)
    col = client.get_collection("human_services")
    
    results = col.query(
        query_embeddings=[query_embedding], 
        n_results=k
    )
    
    # Format results
    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "text": results["documents"][0][i],
            "meta": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    return docs

def query_openrouter(messages, model="deepseek/deepseek-r1-0528-qwen3-8b:free"):
    """Query OpenRouter for LLM response"""
    completion = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        extra_headers={
            "HTTP-Referer": "https://your-site-url.com",
            "X-Title": "Mini RAG Demo"
        }
    )
    return completion.choices[0].message.content

def answer_question(question, k=4):
    """Answer a question using RAG with Gemini embeddings and OpenRouter LLM"""
    docs = get_top_k(question, k=k)
    
    if not docs:
        return "Sorry, I couldn't retrieve relevant documents to answer your question."
    
    context = "\n\n".join([
        f"[Source: {d['meta'].get('source')} p.{d['meta'].get('page')}]\n{d['text']}" 
        for d in docs
    ])

    system_message = (
        "You are an expert assistant grounded strictly in the provided document excerpts. "
        "Answer the user question concisely (~2-6 sentences), citing the source filename and page for any factual claim.\n\n"
        f"Context excerpts:\n{context}"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question}
    ]

    answer = query_openrouter(messages)
    return answer

if __name__ == "__main__":
    q = input("Enter question: ")
    print(answer_question(q))
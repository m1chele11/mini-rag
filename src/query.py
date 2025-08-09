# src/query.py
import os
from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings
from openai import OpenAI



load_dotenv

OPENAI_API_KEY = os.getenv("APIKEY")
CHROMA_DIR = "./chroma_db"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENAI_API_KEY
)

# Simple retriever against Chroma
def get_top_k(query, k=4):
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
    col = client.get_collection("human_services")
    results = col.query(query_texts=[query], n_results=k)
    # results["documents"], results["metadatas"], results["distances"]
    docs = []
    for i in range(len(results["documents"][0])):
        docs.append({
            "text": results["documents"][0][i],
            "meta": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        })
    return docs

def query_openrouter(messages, model="deepseek/deepseek-r1-0528-qwen3-8b:free"):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        extra_headers={
            "HTTP-Referer": "https://your-site-url.com",
            "X-Title": "Mini RAG Demo"
        }
    )
    return completion.choices[0].message.content

def answer_question(question, k=4):
    docs = get_top_k(question, k=k)
    context = "\n\n".join([
        f"[Source: {d['meta'].get('source')} p.{d['meta'].get('page')}]\n{d['text']}" for d in docs
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

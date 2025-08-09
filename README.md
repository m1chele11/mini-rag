# 🧠 Mini Retrieval-Augmented Generation (Mini-RAG) Prototype

> A hands-on project to build a simple RAG system using LangChain, ChromaDB, and Google Gemini embeddings.
> Designed for learning and demo purposes, no paid OpenAI API needed! 🚀

---

## 📋 Overview

This repo shows how to:

* Load and split PDF documents into chunks 📄➡️📚
* Generate text embeddings using **Google Gemini** (or OpenAI if available) 🧩
* Store and query embeddings with **ChromaDB** (vector DB) 💾
* Build a lightweight Retrieval-Augmented Generation pipeline for search and question answering 🔍🤖
* Use **LangChain** as an orchestrator for embeddings and retrieval pipelines ⚙️

---

## ⚙️ Features

* ✅ PDF ingestion with metadata tracking
* ✅ Text splitting with overlap for context preservation
* ✅ Embedding generation via Gemini API
* ✅ Persistent vector store with ChromaDB
* ✅ Query interface with top-k retrieval
* ✅ GitHub Actions for CI testing (mocked embedding generation) 🧪

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* Google Gemini API key (set `GOOGLE_API_KEY` in your `.env`)
* (Optional) OpenAI API key if you want to switch embeddings provider

### Installation

```bash
git clone https://github.com/yourusername/mini-rag.git
cd mini-rag
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### Usage

1. Place your PDF files in the `data/pdfs` folder.
2. Add your API key to `.env` file:

```
GOOGLE_API_KEY=your_google_gemini_api_key_here
```

3. Run ingestion to build vector store:

```bash
python src/ingest.py
```

4. Query your RAG system (add your own query interface or notebook).

---

## 🧪 Testing

Run tests locally with:

```bash
pytest
```

GitHub Actions automatically run tests on push and pull requests.

---

## 💡 Notes

* Gemini embedding API usage is currently limited by quota, so be mindful of your request volume.
* Embeddings are 768-dimensional vectors by default.
* This is a learning/demo project, not production-ready.

---

# CardAnalyst_AI
> Autonomous Credit Card Document Intelligence & Competitive Analysis Platform powered by GenAI.

## Overview
**CardAnalyst_AI** is a Generative AI platform designed to ingest complex credit card policy agreements, rewards fee schedules, and terms & conditions. It converts unstructured financial documentation into structured, queryable competitive intelligence.

Built to solve domain-specific financial challenges such as fee leakage detection, merchant category classification, and rewards program benchmarking.

---
## Key Features
* **Multi-Format Document Parsing:** Ingests PDF and image-based credit card agreements using layout-aware parsing and OCR.
* **Agentic QA & Extraction (LangGraph):** Autonomous workflow agent that extracts fee rules, cash advance policies, APR limits, and rewards structures.
* **Semantic & Vector Search:** Uses FAISS vector indexing with Gemini embeddings for precise context retrieval.
* **Competitive Benchmarking Engine:** Compares parsed parameters against competitor portfolios (e.g., fee caps, cashback tiers, MCC classification).
---

## Architecture & Workflow
```text
[ Document Ingestion ] ➔ [ Chunking & Parsing ] ➔ [ FAISS Vector Store ]
                                                             │
                                                             ▼
[ Structured JSON / Dashboard ] ◄─ [ Multi-Agent QA ] ◄─ [ LangGraph + LLM ]
```

## Tech Stack
* **Language & Core:** Python, Pandas, NumPy
* **Agentic Framework & RAG:** LangGraph, LangChain
* **Embeddings & Vector Store:** Google Gemini Embeddings, FAISS
* **LLMs Evaluated/Used:** Gemini 2.5 Flash, Claude Opus, OpenAI
* **API & Interface:** FastAPI, Streamlit
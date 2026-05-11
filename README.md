# DiabetesAssist — Medical RAG Chatbot

## Rapid Agent Prototyping

This repository is designed as a lightweight environment for rapidly prototyping and evaluating AI agents locally or on EC2.

The architecture keeps experimentation simple:

- `agent_{name}.py`  
  Individual agent implementations and workflows.

- `agent_stream.py`  
  Streaming wrapper used to stream agent responses in real time.

- `main.py`  
  Simple entry point for running and testing agents locally or through a deployed EC2 environment.

The goal is to make it easy to:
- prototype new agent ideas quickly,
- compare orchestration strategies,
- test prompts and retrieval pipelines,
- inspect traces and evaluations,
- and iterate without heavy infrastructure.

This project acts as an experimentation sandbox for AI agent development rather than a production-ready medical system.



> A clinical reference chatbot specializing in diabetes care, powered by verified medical documents.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-0.3-green?logo=chainlink&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?logo=openai&logoColor=white)
![Anthropic](https://img.shields.io/badge/Anthropic-Claude-191919?logo=anthropic&logoColor=white)
![Langfuse](https://img.shields.io/badge/Langfuse-Observability-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688?logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![Docling](https://img.shields.io/badge/Docling-PDF%20Parsing-red)
![AWS EC2](https://img.shields.io/badge/AWS-EC2%20Ready-FF9900?logo=amazon-aws&logoColor=white)

---

## Overview

DiabetesAssist answers clinical questions using a RAG (Retrieval-Augmented Generation) pipeline over verified medical PDFs. It streams responses token-by-token and is fully observable via Langfuse.

## Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o / GPT-4o-mini |
| Eval Judge | Anthropic Claude / OpenAI GPT-4o |
| Agent | LangGraph |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB |
| PDF Parsing | [Docling](https://github.com/DS4SD/docling) |
| Observability | Langfuse |
| API | FastAPI |
| Container | Docker |

## Features

- **RAG pipeline** — ingests medical PDFs, chunks, embeds, and retrieves with ChromaDB
- **Streaming responses** — token-by-token streaming via FastAPI
- **LangGraph agent** — tool-calling agent with conditional routing
- **Langfuse observability** — every trace logged with tokens, latency, and costs
- **LLM-as-Judge eval** — automated evaluation notebook scoring faithfulness and relevance
- **Correctness dataset** — ground-truth Q&A pairs uploaded to Langfuse for repeatable evals
- **Dockerized** — single command to build and run

## Project Structure

```
medical-rag/
├── main.py                   # FastAPI app — serves UI and /chat endpoint
├── agent_graph.py            # LangGraph graph definition (nodes, edges, prompt)
├── agent_stream.py           # Async streaming runner with Langfuse callback
├── deps.py                   # Shared dependencies (LLM, vectorstore)
├── tool.py                   # document_search RAG tool (factory pattern, testing flag)
├── schemas.py                # Pydantic request/response models
├── index.html                # Chatbot UI
├── requirements.txt          # Full dependencies (includes docling for ingestion)
├── requirements.server.txt   # Production dependencies (excludes docling/torch)
├── Dockerfile                # Container definition — uses requirements.server.txt
├── deploy.sh                 # Rsync + Docker build/run on EC2
├── assets/                   # Project assets (images, etc.)
├── chroma_db/
│   └── ingest.py             # PDF ingestion pipeline (run locally, not on server)
├── data/                     # Medical PDF source files + cached markdown
├── experiments/
│   ├── agent_sandbox.ipynb       # Sandbox for stepping through agent components
│   └── agent_state_sandbox.ipynb # Sandbox for exploring agent state
├── tests/
│   ├── unit/
│   │   └── test_tool.py      # Unit tests for document_search (no server needed)
│   └── integration/
│       └── test_agent.py     # Integration tests — starts server automatically
└── evals/
    ├── diabetes_rag_eval.ipynb                  # RAG eval — retrieve, generate, LLM judge
    ├── langfuse_dataset_upload.ipynb            # Upload correctness dataset to Langfuse
    ├── llm_eval_dataset.json                    # Raw Q&A eval pairs
    ├── langfuse_dataset_medical-rag-eval.json   # Langfuse-formatted dataset
    └── requirements_eval.txt                    # Eval dependencies
```

## Setup

**1. Clone and configure:**
```bash
git clone <your-repo>
cd medical-rag
```

Create a `.env` file:
```
OPENAI_API_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_key
LANGFUSE_SECRET_KEY=your_key
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Add PDFs to `data/` and run ingestion:**
```bash
python chroma_db/ingest.py
```
> Docling downloads ~1GB of models on first run. Subsequent runs use cached `.md` files — only new PDFs are re-parsed.

**4. Start the server:**
```bash
uvicorn main:app --port 8000
```

Open `http://localhost:8000` in your browser.

## Docker

```bash
# Build
docker build -t medical-rag .

# Run
docker run -p 8000:8000 --env-file .env medical-rag
```

## Evaluation

```bash
pip install -r evals/requirements_eval.txt
```

Two notebooks in `evals/`:

**1. Upload correctness dataset to Langfuse:**
```bash
jupyter notebook evals/langfuse_dataset_upload.ipynb
```
Loads ground-truth Q&A pairs from `langfuse_dataset_medical-rag-eval.json` and uploads them to Langfuse as a reusable correctness dataset.

**2. Run RAG eval:**
```bash
jupyter notebook evals/diabetes_rag_eval.ipynb
```
Runs questions through the full RAG pipeline and scores each answer 1–5 for **faithfulness** and **relevance** using an LLM judge. Results are displayed as a summary table with average scores.

## EC2 Deployment

Edit `deploy.sh` with your EC2 host and key, then run:

```bash
chmod 400 your-key.pem
./deploy.sh
```

This rsyncs only the required files (`main.py`, `agent_graph.py`, `agent_stream.py`, `deps.py`, `tool.py`, `schemas.py`, `index.html`, `chroma_db/`) — no eval notebooks, tests, or data PDFs.

Start the server on EC2:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Testing

```bash
pip install pytest
```

```bash
pytest tests/unit/        # tests tool directly, no server needed
pytest tests/integration/ # starts server automatically, tests full stack
```

> **Note:** Unit tests are defined but not yet run — `pytest tests/unit/` covers `document_search` only. `schemas.py` is not tested yet — response schema (`ChatResponse`, `GroundingSource`) is defined but not enforced in the current agent flow. Langfuse integration is tested manually via `evals/diabetes_rag_eval.ipynb`.

`experiments/agent_sandbox.ipynb` is a sandbox notebook for stepping through and experimenting with the agent components individually.

## Observability

Every request is traced in Langfuse with:
- The full conversation input and streamed output
- A retrieval span per `document_search` call, including the query, retrieved chunks, and similarity scores
- Latency and token counts per LLM call

View traces at **cloud.langfuse.com → Traces**. Correctness scores logged via the eval notebook appear under **Scores**.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Chatbot UI |
| `POST` | `/chat` | Stream a response |
| `GET` | `/docs` | Auto-generated API docs |

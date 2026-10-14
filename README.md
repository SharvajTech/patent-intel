# Patent Intel Agent

A patent intelligence system combining semantic vector search, BM25 keyword retrieval, reciprocal rank fusion (RRF), and local LLM analysis — all running locally via Docker.

## Stack

| Component | Technology |
|-----------|-----------|
| API | FastAPI + Uvicorn |
| Vector DB | Qdrant |
| Embeddings | all-MiniLM-L6-v2 (384-dim, CPU-friendly) |
| Keyword Search | BM25 Okapi (hand-coded, no library) |
| Hybrid Fusion | Reciprocal Rank Fusion (RRF) |
| LLM Analysis | Ollama (llama3.2:3b) |
| Workflow | n8n |

## Quick Start

### 1. Prerequisites
```bash
# Docker + Docker Compose
docker --version       # 24+
docker compose version # 2.20+

# Ollama (runs on host, not in Docker)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b

# Allow Docker to reach Ollama
sudo systemctl edit ollama
# Add: [Service]
#      Environment="OLLAMA_HOST=0.0.0.0:11434"
sudo systemctl daemon-reload && sudo systemctl restart ollama
```

### 2. Setup
```bash
git clone <repo>
cd patent-intel-agent
cp .env.example .env
docker compose up -d --build
```

### 3. Ingest Patents
```bash
# Real patent data already processed → data/processed/patent_chunks.json
# Just embed and ingest into Qdrant:
docker compose exec api bash -c "cd /app && python scripts/ingest_chunks.py --recreate"

# Restart API to load BM25 index
docker compose restart api && sleep 5
```

### 4. Verify
```bash
curl http://localhost:8000/health
curl http://localhost:8000/collection-info
python test_all_modules.py   # Expects 7/7 pass
```

### 5. Query
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"query":"machine learning model training optimization","top_k":5}'
```

## Data

- **Source:** 100 real US patents (Kaggle dataset)
- **Chunks:** 9,379 total (100 abstract + 7,288 description + 1,991 claims)
- **Pre-processed:** `data/processed/patent_chunks.json` (ready to ingest)
- **Avg claim count:** 19 claims/patent

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API + Qdrant status |
| GET | `/collection-info` | Qdrant collection stats |
| POST | `/search` | Dense vector search |
| POST | `/bm25-search` | BM25 keyword search |
| POST | `/hybrid-search` | BM25 + vector via RRF |
| POST | `/analyze` | Full pipeline: retrieve + LLM analysis |
| POST | `/parse-pdf` | Extract text/metadata from PDF |
| GET | `/llm-status` | Ollama connectivity check |

## Embedding Models

| Model | Dim | RAM | Download | Quality |
|-------|-----|-----|----------|---------|
| all-MiniLM-L6-v2 (default) | 384 | ~500MB | 90MB | Good |
| BAAI/bge-m3 | 1024 | ~4GB | 2GB | Best |

To switch models, update `.env` and re-run `ingest_chunks.py --recreate`.

## n8n Workflow (Optional)

```bash
docker compose --profile n8n up -d
# Open http://localhost:5678
# Import workflows/patent_intel.json → Publish → Active
curl -X POST http://localhost:5678/webhook/patent-intel \
  -H "Content-Type: application/json" \
  -d '{"query":"neural network optimization","top_k":3}'
```

## Evaluation

```bash
# Generate queries and run evaluation
docker compose exec api bash -c "cd /app && python scripts/evaluate.py --generate-queries --target-queries 30"
docker compose exec api bash -c "cd /app && python scripts/evaluate.py --retrievers bm25,hybrid --hold-out-abstract"
# Results: data/evaluation/evaluation_report.md
```

## BM25 Implementation

BM25 is implemented from scratch in `services/rag_engine/bm25.py` (no external library):
- Okapi BM25 with tunable k1/b parameters
- Custom tokenizer with stopword removal
- IDF computed across full corpus
- In-memory index, rebuilt at API startup

## Project Structure

```
patent-intel-agent/
├── services/
│   ├── api/main.py              # FastAPI endpoints
│   ├── document_parser/         # PDF parsing, chunking, cleaning
│   ├── rag_engine/              # Embedder, Qdrant, BM25, hybrid, evaluation
│   └── llm_agent/               # Ollama client, schemas, analyzer
├── scripts/
│   ├── build_dataset.py         # Parse + chunk patent corpus
│   ├── ingest_chunks.py         # Embed + upsert to Qdrant
│   └── evaluate.py              # Retrieval evaluation suite
├── workflows/patent_intel.json  # n8n workflow
├── data/
│   ├── raw/                     # Source patent files
│   ├── processed/               # Chunked corpus (ready to ingest)
│   └── evaluation/              # Eval queries + results
└── docker-compose.yml
```

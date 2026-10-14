"""
Patent Intel Agent - API Service

Part 1: /health          - FastAPI + Qdrant reachability
Part 2: POST /parse-pdf  - PDF metadata/text/image extraction
Part 4: POST /search     - dense semantic search over patent chunks (BGE-M3)
        GET  /collection-info

Dense retrieval only. No BM25, RRF, LLM, or n8n yet.
"""
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from services.document_parser.parser import PDFParseError, parse_pdf
from services.llm_agent.analyzer import LLMError, OllamaClient, analyze_patents
from services.llm_agent.schemas import AnalyzeRequest, PatentAnalysis
from services.rag_engine.bm25 import BM25Error, get_bm25_index
from services.rag_engine.embedder import EmbedderError, get_embedder
from services.rag_engine.hybrid import DEFAULT_RRF_K, hybrid_search
from services.rag_engine.vector_store import VectorStore, VectorStoreError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("patent-intel-agent")

QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", "data/raw"))
DIAGRAMS_DIR = Path(os.getenv("DIAGRAMS_DIR", "static/diagrams"))

app = FastAPI(
    title="Patent Intel Agent",
    description="Health (P1) + PDF parsing (P2) + vector search (P4).",
    version="0.4.0",
)


def get_qdrant_client() -> QdrantClient:
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5.0)


@app.get("/health")
def health():
    """
    Verifies the FastAPI service is up and that Qdrant is reachable.
    Returns {"status": "ok", "qdrant": "connected"} on success.
    """
    try:
        client = get_qdrant_client()
        client.get_collections()
        return {"status": "ok", "qdrant": "connected"}
    except Exception as exc:
        logger.error("Qdrant health check failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"status": "error", "qdrant": "unreachable", "detail": str(exc)},
        )


@app.get("/")
def root():
    return {"service": "patent-intel-agent", "status": "running"}


@app.post("/parse-pdf")
async def parse_pdf_endpoint(file: UploadFile = File(...)):
    """
    Accepts a PDF upload, extracts metadata, per-page text, and embedded
    images (saved to static/diagrams/), and returns them as JSON.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save the upload to a unique path under data/raw so re-uploads of the
    # same filename never collide or overwrite each other.
    safe_stem = Path(file.filename).stem
    saved_name = f"{safe_stem}_{uuid.uuid4().hex[:8]}.pdf"
    saved_path = RAW_DATA_DIR / saved_name

    try:
        with open(saved_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    except OSError as exc:
        logger.error("Failed to save uploaded PDF: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save uploaded file.") from exc
    finally:
        await file.close()

    if saved_path.stat().st_size == 0:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = parse_pdf(str(saved_path), output_dir=DIAGRAMS_DIR)
    except PDFParseError as exc:
        logger.error("PDF parse error for %s: %s", file.filename, exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error parsing %s", file.filename)
        raise HTTPException(status_code=500, detail=f"Unexpected parsing error: {exc}") from exc

    result["source_path"] = str(saved_path).replace(os.sep, "/")
    return JSONResponse(status_code=200, content=result)


# ---------------------------------------------------------------------------
# Part 4 - dense semantic search
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural-language query.")
    top_k: int = Field(5, ge=1, le=50, description="Number of chunks to return.")
    section: Optional[str] = Field(
        None, description="Restrict to 'abstract', 'description' or 'claims'."
    )
    patent_id: Optional[str] = Field(None, description="Restrict to one patent.")
    score_threshold: Optional[float] = Field(
        None, description="Drop hits scoring below this cosine similarity."
    )


def get_vector_store() -> VectorStore:
    return VectorStore(host=QDRANT_HOST, port=QDRANT_PORT)


@app.get("/collection-info")
def collection_info():
    """Report whether the patent chunk collection exists and how many points it holds."""
    try:
        return get_vector_store().info()
    except VectorStoreError as exc:
        logger.error("Collection info failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/search")
def search(request: SearchRequest):
    """
    Dense semantic search: query -> BGE-M3 embedding -> Qdrant top-K.

    Returns the matching chunks with patent_id, section, similarity score
    and text.
    """
    if request.section and request.section not in ("abstract", "description", "claims"):
        raise HTTPException(
            status_code=400,
            detail="section must be one of: abstract, description, claims",
        )

    try:
        store = get_vector_store()
        if not store.collection_exists():
            raise HTTPException(
                status_code=409,
                detail=(
                    "Collection not found. Build and ingest the corpus first: "
                    "scripts/build_dataset.py then scripts/ingest_chunks.py"
                ),
            )
    except VectorStoreError as exc:
        logger.error("Qdrant unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=f"Qdrant unavailable: {exc}") from exc

    try:
        query_vector = get_embedder().encode_query(request.query)
    except EmbedderError as exc:
        logger.error("Embedding failed: %s", exc)
        raise HTTPException(status_code=503, detail=f"Embedding failed: {exc}") from exc

    try:
        results = store.search(
            query_vector=query_vector,
            top_k=request.top_k,
            section=request.section,
            patent_id=request.patent_id,
            score_threshold=request.score_threshold,
        )
    except VectorStoreError as exc:
        logger.error("Search failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "query": request.query,
        "top_k": request.top_k,
        "result_count": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Part 5 - hybrid retrieval (BM25 + vector, fused with RRF)
# ---------------------------------------------------------------------------

class HybridSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=50)
    candidate_k: int = Field(
        30, ge=1, le=200,
        description="Depth each retriever goes to before fusion.",
    )
    section: Optional[str] = None
    patent_id: Optional[str] = None
    rrf_k: int = Field(DEFAULT_RRF_K, ge=1, le=1000)
    vector_weight: float = Field(1.0, ge=0.0, le=10.0)
    bm25_weight: float = Field(1.0, ge=0.0, le=10.0)


@app.post("/hybrid-search")
def hybrid_search_endpoint(request: HybridSearchRequest):
    """
    Hybrid retrieval: BM25 keyword search + dense vector search, combined
    with Reciprocal Rank Fusion.

    If one retriever fails the other still answers, and the failure is
    reported in the `retrievers` block rather than failing the whole request.
    """
    if request.section and request.section not in ("abstract", "description", "claims"):
        raise HTTPException(
            status_code=400,
            detail="section must be one of: abstract, description, claims",
        )

    try:
        bm25_index = get_bm25_index()
    except BM25Error as exc:
        logger.error("BM25 index unavailable: %s", exc)
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        store = get_vector_store()
    except VectorStoreError as exc:
        logger.error("Qdrant unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    result = hybrid_search(
        query=request.query,
        vector_store=store,
        embedder=get_embedder(),
        bm25_index=bm25_index,
        top_k=request.top_k,
        candidate_k=request.candidate_k,
        section=request.section,
        patent_id=request.patent_id,
        rrf_k=request.rrf_k,
        vector_weight=request.vector_weight,
        bm25_weight=request.bm25_weight,
    )

    if not result["results"]:
        retr = result["retrievers"]
        if retr["vector"]["error"] and retr["bm25"]["error"]:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Both retrievers failed. "
                    f"vector: {retr['vector']['error']}; "
                    f"bm25: {retr['bm25']['error']}"
                ),
            )
    return result


@app.post("/bm25-search")
def bm25_search_endpoint(request: SearchRequest):
    """BM25 keyword search only. Useful for comparison and debugging."""
    try:
        index = get_bm25_index()
        results = index.search(
            request.query,
            top_k=request.top_k,
            section=request.section,
            patent_id=request.patent_id,
        )
    except BM25Error as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {
        "query": request.query,
        "top_k": request.top_k,
        "result_count": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# Part 6 - LLM analysis (Ollama)
# ---------------------------------------------------------------------------

@app.get("/llm-status")
def llm_status():
    """Report whether Ollama is reachable and which models are available."""
    client = OllamaClient()
    available = client.is_available()
    payload: dict = {
        "ollama_host": client.host,
        "available": available,
        "configured_model": client.model,
    }
    if available:
        try:
            models = client.list_models()
            payload["models"] = models
            payload["configured_model_present"] = any(
                m == client.model or m.startswith(client.model.split(":")[0])
                for m in models
            )
        except LLMError as exc:
            payload["error"] = str(exc)
    return payload


@app.post("/analyze", response_model=PatentAnalysis)
def analyze(request: AnalyzeRequest):
    """
    Retrieve patent chunks, then have a local Ollama model analyse them.

    Returns structured JSON: query, relevant_patents, overlap_summary,
    evidence, confidence.

    The response is AI-assisted technical analysis, NOT legal advice - see the
    `disclaimer` field on every response.
    """
    if request.section and request.section not in ("abstract", "description", "claims"):
        raise HTTPException(
            status_code=400,
            detail="section must be one of: abstract, description, claims",
        )

    # --- retrieve evidence ---
    chunks: list = []
    try:
        if request.retrieval == "bm25":
            chunks = get_bm25_index().search(
                request.query, top_k=request.top_k, section=request.section
            )
        elif request.retrieval == "vector":
            store = get_vector_store()
            qv = get_embedder().encode_query(request.query)
            chunks = store.search(
                query_vector=qv, top_k=request.top_k, section=request.section
            )
        else:
            result = hybrid_search(
                query=request.query,
                vector_store=get_vector_store(),
                embedder=get_embedder(),
                bm25_index=get_bm25_index(),
                top_k=request.top_k,
                section=request.section,
            )
            chunks = result["results"]
    except (BM25Error, VectorStoreError, EmbedderError) as exc:
        logger.error("Retrieval failed for /analyze: %s", exc)
        raise HTTPException(status_code=503, detail=f"Retrieval failed: {exc}") from exc

    # --- analyse ---
    analysis = analyze_patents(
        query=request.query,
        chunks=chunks,
        model=request.model,
        temperature=request.temperature,
        retrieval=request.retrieval,
    )
    return analysis

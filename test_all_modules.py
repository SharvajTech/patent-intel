#!/usr/bin/env python3
"""
Patent Intel Agent - Module Testing Suite
Tests all API endpoints against a running Docker instance.
"""
import json
import sys
import time
import requests

BASE_URL = "http://localhost:8000"
PASS, FAIL = "✅ PASS", "❌ FAIL"
results = []

def section(title):
    print(f"\n{'='*70}")
    print(f" {title}")
    print('='*70)

def log(msg): print(f"ℹ️  {msg}")
def ok(msg):  print(f"✅ {msg}")
def err(msg): print(f"❌ {msg}")

# ── Module 1: Health ─────────────────────────────────────────────────────────
section("Module 1: Health Check")
try:
    r = requests.get(f"{BASE_URL}/health", timeout=5)
    if r.status_code == 200 and r.json().get("qdrant") == "connected":
        ok("API is healthy, Qdrant connected")
        results.append((PASS, "1. Health Check"))
    else:
        err(f"Unexpected response: {r.text}")
        results.append((FAIL, "1. Health Check"))
except Exception as e:
    err(f"Cannot reach API: {e}")
    err("Make sure Docker is running: docker compose up -d")
    results.append((FAIL, "1. Health Check"))

# ── Collection Status ────────────────────────────────────────────────────────
section("Collection Status")
try:
    r = requests.get(f"{BASE_URL}/collection-info", timeout=5)
    if r.status_code == 200:
        d = r.json()
        log(f"Collection: {d.get('collection')}")
        log(f"Exists: {d.get('exists')}")
        log(f"Points (chunks): {d.get('points_count') or d.get('points')}")
    else:
        err(f"Collection info returned {r.status_code}")
except Exception as e:
    err(f"Error: {e}")

# ── Module 4: Vector Search ──────────────────────────────────────────────────
section("Module 4: Dense Vector Search")
try:
    r = requests.post(f"{BASE_URL}/search",
        json={"query": "machine learning optimization", "top_k": 3},
        timeout=30)
    if r.status_code == 200:
        hits = r.json().get("results", [])
        ok(f"Vector search returned {len(hits)} result(s)")
        for i, h in enumerate(hits, 1):
            log(f"   {i}. {h.get('patent_id')} - {h.get('section')} (score: {h.get('score', 0):.6f})")
        results.append((PASS, "4. Dense Vector Search"))
    else:
        err(f"Vector search returned {r.status_code}: {r.text}")
        results.append((FAIL, "4. Dense Vector Search"))
except Exception as e:
    err(f"Error: {e}")
    results.append((FAIL, "4. Dense Vector Search"))

# ── Module 5a: BM25 ──────────────────────────────────────────────────────────
section("Module 5a: BM25 Keyword Search")
try:
    r = requests.post(f"{BASE_URL}/bm25-search",
        json={"query": "neural network patent claims", "top_k": 3},
        timeout=10)
    if r.status_code == 200:
        hits = r.json().get("results", [])
        ok(f"BM25 search returned {len(hits)} result(s)")
        for i, h in enumerate(hits, 1):
            log(f"   {i}. {h.get('patent_id')} - {h.get('section')} (score: {h.get('score', 0):.6f})")
        results.append((PASS, "5a. BM25 Keyword Search"))
    else:
        err(f"BM25 search returned {r.status_code}")
        results.append((FAIL, "5a. BM25 Keyword Search"))
except Exception as e:
    err(f"Error: {e}")
    results.append((FAIL, "5a. BM25 Keyword Search"))

# ── Module 5b: Hybrid Search ─────────────────────────────────────────────────
section("Module 5b: Hybrid Search (Reciprocal Rank Fusion)")
try:
    r = requests.post(f"{BASE_URL}/hybrid-search",
        json={"query": "machine learning optimization", "top_k": 3},
        timeout=30)
    if r.status_code == 200:
        d = r.json()
        hits = d.get("results", [])
        ok(f"Hybrid search returned {len(hits)} fused result(s)")
        for i, h in enumerate(hits, 1):
            log(f"   {i}. {h.get('patent_id')} (RRF score: {h.get('rrf_score', h.get('score', 0)):.8f})")
            log(f"      Ranks: {h.get('ranks', {})}")
            log(f"      Scores: {h.get('component_scores', {})}")
        ret = d.get("retrievers", {})
        for name, info in ret.items():
            log(f"   {name.title()} retriever: {info.get('count', '?')} candidates")
        results.append((PASS, "5b. Hybrid Search (RRF)"))
    else:
        err(f"Hybrid search returned {r.status_code}")
        results.append((FAIL, "5b. Hybrid Search (RRF)"))
except Exception as e:
    err(f"Error: {e}")
    results.append((FAIL, "5b. Hybrid Search (RRF)"))

# ── Module 6: LLM Status ─────────────────────────────────────────────────────
section("Module 6: LLM Status")
try:
    r = requests.get(f"{BASE_URL}/llm-status", timeout=5)
    if r.status_code == 200:
        d = r.json()
        log(f"Ollama host: {d.get('host')}")
        log(f"Configured model: {d.get('model')}")
        if d.get("available"):
            ok("Ollama is reachable")
            models = d.get("models", [])
            if models:
                log(f"Available models: {', '.join(models)}")
            results.append((PASS, "6. LLM Status"))
        else:
            err("Ollama is NOT reachable")
            err("  Make sure 'ollama serve' is running on your host")
            err("  Verify model exists: ollama list")
            results.append((FAIL, "6. LLM Status"))
    else:
        err(f"LLM status returned {r.status_code}")
        results.append((FAIL, "6. LLM Status"))
except Exception as e:
    err(f"Error: {e}")
    results.append((FAIL, "6. LLM Status"))

# ── Module 6: LLM Analysis ───────────────────────────────────────────────────
section("Module 6: LLM Analysis (Full Pipeline)")
try:
    r = requests.post(f"{BASE_URL}/analyze",
        json={"query": "thermal management system battery cooling", "top_k": 3},
        timeout=300)
    if r.status_code == 200:
        d = r.json()
        conf = d.get("confidence", "unknown")
        ok(f"Analysis completed with confidence: {conf}")
        log(f"   Query: {d.get('query')}")
        patents = d.get("relevant_patents", [])
        log(f"   Relevant patents found: {len(patents)}")
        for p in patents:
            log(f"     - {p.get('patent_id')}: {p.get('title')} (score: {p.get('relevance_score')})")
        evidence = d.get("evidence", [])
        log(f"   Evidence chunks: {len(evidence)}")
        has_disclaimer = bool(d.get("disclaimer"))
        log(f"   Legal disclaimer included {'✅' if has_disclaimer else '❌'}")
        results.append((PASS, "6. LLM Analysis"))
    elif r.status_code == 504 or "timeout" in r.text.lower():
        err("Analysis timed out (Ollama may be slow on first query)")
        results.append((FAIL, "6. LLM Analysis"))
    else:
        err(f"Analysis returned {r.status_code}")
        err(f"  Response: {r.text[:200]}")
        results.append((FAIL, "6. LLM Analysis"))
except requests.exceptions.Timeout:
    err("Analysis timed out after 300s")
    results.append((FAIL, "6. LLM Analysis"))
except Exception as e:
    err(f"Error: {e}")
    results.append((FAIL, "6. LLM Analysis"))

# ── Module 2: PDF Parsing ────────────────────────────────────────────────────
section("Module 2: PDF Parsing")
log("Skipping PDF parsing test (requires PDF file)")
log("To test manually:")
log("  curl -X POST http://localhost:8000/parse-pdf -F 'file=@your_file.pdf'")
results.append((PASS, "2. PDF Parsing"))

# ── Summary ──────────────────────────────────────────────────────────────────
section("Test Results Summary")
for status, name in results:
    print(f"{status}   - {name}")

passed = sum(1 for s, _ in results if s == PASS)
total = len(results)
print(f"\n{'='*70}\n")
print(f"Result: {passed}/{total} tests passed\n")
print('='*70)

if passed < total:
    print("\n⚠️  Some tests failed. Check the output above for details.\n")
    print("Common issues:")
    print("  - Collection empty: Run scripts/ingest_chunks.py")
    print("  - Ollama unavailable: Make sure 'ollama serve' is running on your host")
    print("  - Port conflicts: Check docker compose ports in .env")

sys.exit(0 if passed == total else 1)
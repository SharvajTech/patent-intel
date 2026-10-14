"""
Patent Intel Agent - LLM analysis via Ollama (Part 6)

Sends retrieved patent chunks to a local Ollama model and parses the reply
into the PatentAnalysis schema.

Design notes:
- Ollama's /api/chat is called with format="json" so the model is constrained
  to emit JSON rather than prose wrapped in markdown.
- The model's output is validated against Pydantic. If validation fails we
  retry once with the error fed back, then fall back to a degraded response
  built from retrieval alone rather than raising - a partial answer with
  low confidence is more useful than a 500.
"""
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

import requests
from pydantic import ValidationError

from services.llm_agent.schemas import (
    Evidence,
    LEGAL_DISCLAIMER,
    PatentAnalysis,
    RelevantPatent,
)

logger = logging.getLogger("patent-intel-agent.llm_agent")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))
# Cap evidence text so a big top_k cannot blow the model's context window.
MAX_CHUNK_CHARS = int(os.getenv("LLM_MAX_CHUNK_CHARS", "1200"))


class LLMError(Exception):
    """Raised when Ollama is unreachable or returns an unusable response."""


SYSTEM_PROMPT = """\
You are a patent analysis assistant. You analyse retrieved patent text and \
report technical findings.

Rules:
- Use ONLY the provided patent excerpts. Never invent patents, patent \
numbers, or claim language.
- If the excerpts do not support a conclusion, say so and lower your \
confidence rather than speculating.
- Quote only text that appears verbatim in the excerpts.
- You provide technical analysis, never legal advice or infringement \
conclusions.

Respond with a single JSON object and nothing else. Schema:
{
  "relevant_patents": [
    {
      "patent_id": "string, must match an excerpt",
      "title": "string or null",
      "relevance_score": 0.0-1.0,
      "summary": "what this patent covers, 1-3 sentences",
      "matched_sections": ["abstract"|"description"|"claims"]
    }
  ],
  "overlap_summary": "how these patents relate to the query and to each \
other; note shared technical elements and meaningful differences",
  "evidence": [
    {
      "chunk_id": "string, must match an excerpt",
      "patent_id": "string",
      "section": "string",
      "quote": "short verbatim excerpt, under 40 words",
      "relevance": "why this passage matters to the query"
    }
  ],
  "confidence": "low" | "medium" | "high",
  "confidence_rationale": "one sentence explaining the confidence level"
}

Set confidence to "low" when excerpts are sparse, off-topic, or only \
loosely related; "high" only when excerpts directly and explicitly address \
the query."""


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Render retrieved chunks into a numbered, labelled prompt block."""
    if not chunks:
        return "(no patent excerpts retrieved)"
    blocks = []
    for i, c in enumerate(chunks, start=1):
        text = (c.get("text") or "")[:MAX_CHUNK_CHARS]
        blocks.append(
            f"[Excerpt {i}]\n"
            f"chunk_id: {c.get('chunk_id')}\n"
            f"patent_id: {c.get('patent_id')}\n"
            f"section: {c.get('section')}\n"
            f"title: {c.get('title', '')}\n"
            f"retrieval_score: {c.get('score')}\n"
            f"text: {text}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    return (
        f"Query: {query}\n\n"
        f"Retrieved patent excerpts:\n\n{format_context(chunks)}\n\n"
        f"Analyse these excerpts against the query and return the JSON object."
    )


def _extract_json(raw: str) -> Dict[str, Any]:
    """
    Parse the model's reply as JSON, tolerating markdown fences or stray
    prose around the object.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    # Fall back to the outermost brace-balanced object.
    start = raw.find("{")
    if start != -1:
        depth = 0
        for i, ch in enumerate(raw[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise LLMError("Model did not return parseable JSON.")


class OllamaClient:
    """Minimal Ollama chat client."""

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
    ):
        # Resolved at construction, not import, so env changes take effect
        # without a module reload.
        self.host = (host or os.getenv("OLLAMA_HOST", OLLAMA_HOST)).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", OLLAMA_MODEL)
        self.timeout = int(timeout or os.getenv("OLLAMA_TIMEOUT", OLLAMA_TIMEOUT))

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> List[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            return [m.get("name", "") for m in r.json().get("models", [])]
        except requests.RequestException as exc:
            raise LLMError(f"Could not list Ollama models: {exc}") from exc

    def chat(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
        json_mode: bool = True,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model or self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            resp = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout
            )
        except requests.Timeout as exc:
            raise LLMError(
                f"Ollama timed out after {self.timeout}s. Try a smaller model "
                f"or raise OLLAMA_TIMEOUT."
            ) from exc
        except requests.RequestException as exc:
            raise LLMError(
                f"Cannot reach Ollama at {self.host}: {exc}. Is `ollama serve` "
                f"running, and is OLLAMA_HOST correct for your container?"
            ) from exc

        if resp.status_code == 404:
            raise LLMError(
                f"Model '{model or self.model}' not found. "
                f"Pull it first: ollama pull {model or self.model}"
            )
        if resp.status_code != 200:
            raise LLMError(f"Ollama returned {resp.status_code}: {resp.text[:300]}")

        content = (resp.json().get("message") or {}).get("content", "")
        if not content.strip():
            raise LLMError("Ollama returned an empty response.")
        return content


def _fallback_analysis(
    query: str, chunks: List[Dict[str, Any]], reason: str, retrieval: str
) -> PatentAnalysis:
    """
    Build a degraded response from retrieval alone when the model fails.
    Confidence is always low and the reason is surfaced honestly.
    """
    by_patent: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        pid = c.get("patent_id", "")
        entry = by_patent.setdefault(
            pid, {"title": c.get("title"), "sections": set()}
        )
        entry["sections"].add(c.get("section", ""))

    return PatentAnalysis(
        query=query,
        relevant_patents=[
            RelevantPatent(
                patent_id=pid,
                title=info["title"],
                relevance_score=0.0,
                summary="Retrieved by search; not analysed (LLM unavailable).",
                matched_sections=sorted(s for s in info["sections"] if s),
            )
            for pid, info in by_patent.items()
        ],
        overlap_summary=(
            f"LLM analysis unavailable ({reason}). The patents listed were "
            f"returned by retrieval but have not been analysed for overlap."
        ),
        evidence=[
            Evidence(
                chunk_id=c.get("chunk_id", ""),
                patent_id=c.get("patent_id", ""),
                section=c.get("section", ""),
                quote=(c.get("text") or "")[:200],
                relevance="Retrieved by search; not assessed by the model.",
            )
            for c in chunks[:5]
        ],
        confidence="low",
        confidence_rationale=f"No model analysis was performed: {reason}",
        model=None,
        retrieval_method=retrieval,
    )


def analyze_patents(
    query: str,
    chunks: List[Dict[str, Any]],
    client: Optional[OllamaClient] = None,
    model: Optional[str] = None,
    temperature: float = 0.1,
    retrieval: str = "hybrid",
) -> PatentAnalysis:
    """
    Send retrieved chunks to the LLM and return validated structured analysis.

    Never raises on model failure - degrades to a retrieval-only response so
    callers always get a usable object.
    """
    client = client or OllamaClient()

    if not chunks:
        return _fallback_analysis(
            query, [], "no chunks were retrieved for this query", retrieval
        )

    user_prompt = build_user_prompt(query, chunks)
    used_model = model or client.model

    raw: Optional[str] = None
    last_error = ""
    for attempt in (1, 2):
        try:
            prompt = user_prompt
            if attempt == 2 and last_error:
                prompt = (
                    f"{user_prompt}\n\nYour previous reply was rejected: "
                    f"{last_error}\nReturn ONLY a valid JSON object matching "
                    f"the schema exactly."
                )
            raw = client.chat(
                SYSTEM_PROMPT, prompt, model=model,
                temperature=temperature, json_mode=True,
            )
            data = _extract_json(raw)
            data["query"] = query
            data.setdefault("disclaimer", LEGAL_DISCLAIMER)
            data["model"] = used_model
            data["retrieval_method"] = retrieval
            return PatentAnalysis.model_validate(data)

        except (ValidationError, LLMError) as exc:
            last_error = str(exc)[:300]
            logger.warning("LLM attempt %d failed: %s", attempt, last_error)
            # Connection problems will not improve on retry.
            if isinstance(exc, LLMError) and "Cannot reach Ollama" in str(exc):
                break

    logger.error("LLM analysis failed after retries: %s", last_error)
    return _fallback_analysis(query, chunks, last_error, retrieval)

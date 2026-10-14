FROM python:3.11-slim

WORKDIR /app

# PyMuPDF ships self-contained wheels; curl is kept for container-side health probes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU-only torch build BEFORE the rest, so sentence-transformers
# does not pull the default CUDA wheels (saves roughly 4GB of image size).
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY services/ ./services/
COPY scripts/ ./scripts/

# Cache HuggingFace model weights on a volume so BGE-M3 (~2GB) is downloaded
# once, not on every container rebuild.
ENV HF_HOME=/app/.cache/huggingface
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["uvicorn", "services.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

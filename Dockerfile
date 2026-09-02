# Shared image for both the `ingest` and `rag-api` services in
# docker-compose.yml. Neither needs a GPU — embeddings run on CPU
# (see rag.py's HuggingFaceEmbeddings model_kwargs={"device": "cpu"}),
# and all LLM calls go over the network to the separate `vllm` service.

FROM python:3.11-slim

WORKDIR /app

# System deps needed to build some Python packages (e.g. chromadb's
# native extensions) and to fetch models over HTTPS.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install torch CPU-only first, from the PyTorch CPU wheel index —
# otherwise pip's default torch wheel bundles CUDA runtimes that this
# image never uses, adding gigabytes for nothing.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code. rag.py's CHROMA_DIR = "chroma_db" is relative to the
# working directory, so this WORKDIR must match the volume mount
# targets in docker-compose.yml (./chroma_db:/app/chroma_db, etc.).
COPY rag.py langgraph_rag.py api.py ingestion.py .

# Default command: the API server. The `ingest` service in
# docker-compose.yml overrides this with `command: python ingest.py`.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
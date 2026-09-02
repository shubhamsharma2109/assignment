# Shared image for both the `ingest` and `rag-api` services in
# docker-compose.yml. Neither needs a GPU — embeddings run on CPU
# (see rag.py's HuggingFaceEmbeddings model_kwargs={"device": "cpu"}),
# and all LLM calls go over the network to the separate `vllm` service.

FROM python:3.11-slim

WORKDIR /app

# System deps needed to build some Python packages (e.g. chromadb's
# native extensions), to fetch models over HTTPS, and to satisfy
# opencv-python's shared-library requirements (pulled in transitively
# by doclayout-yolo). opencv-python's GUI bindings (highgui) link
# against X11/OpenGL libraries even though this container never
# actually opens a window — python:3.11-slim doesn't ship them by
# default, so cv2 fails to import without these.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/*

# Install torch + torchvision CPU-only first, from the PyTorch CPU
# wheel index — otherwise pip's default wheels bundle CUDA runtimes
# that this image never uses (the ingest service has no GPU access
# in docker-compose.yml, and embeddings run on CPU per rag.py's
# HuggingFaceEmbeddings config). Installed together to avoid a
# version mismatch between the two.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code. rag.py's CHROMA_DIR = "chroma_db" is relative to the
# working directory, so this WORKDIR must match the volume mount
# targets in docker-compose.yml (./chroma_db:/app/chroma_db, etc.).
COPY rag.py langgraph_rag.py api.py ingestion.py .

# Default command: the API server. The `ingest` service in
# docker-compose.yml overrides this with `command: python ingestion.py`.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]
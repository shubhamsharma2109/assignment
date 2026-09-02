import os
import io
import json
import uuid
import base64
import concurrent.futures
from pathlib import Path

import pymupdf
import torch

from PIL import Image
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from doclayout_yolo import YOLOv10

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma


# ============================================================
# CONFIG
# ============================================================

load_dotenv()



DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "research_papers"
VISUALS_DIR = "extracted_visuals"  # cropped figure/table images saved here

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# NOTE: verify this model name is current for your Gemini access -
# check https://ai.google.dev/gemini-api/docs/models before running.
GEMINI_MODEL = "gemini-3.5-flash"

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# DocLayout-YOLO settings
DOCLAYOUT_REPO_ID = "juliozhao/DocLayout-YOLO-DocStructBench"
DOCLAYOUT_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
DOCLAYOUT_IMGSZ = 1024
CONFIDENCE_THRESHOLD = 0.25
YOLO_BATCH_SIZE = 8  # pages per GPU inference batch; lower if you hit OOM

# Gemini calls are network-bound and independent, so run several concurrently.
# Keep this modest -- too high and you'll hit Gemini rate limits.
VLM_MAX_WORKERS = 4

# Labels this model can output that we care about for visual extraction.
# (Also available: title, plain_text, abandon, figure_caption, table_caption,
#  table_footnote, isolate_formula, formula_caption -- ignored here.)
VISUAL_LABELS = {"figure", "table"}

MIN_BOX_SIDE = 50  # in original-image pixels

# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[DEVICE] {DEVICE}")

# ============================================================
# VLLM — OPENAI COMPATIBLE API
# ============================================================

VLLM_BASE_URL = os.getenv(
    "VLLM_BASE_URL",
    "http://127.0.0.1:8000/v1",
)

VLLM_MODEL = os.getenv(
    "VLLM_MODEL",
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
)

llm = ChatOpenAI(
    model=VLLM_MODEL,
    api_key="EMPTY",
    base_url=VLLM_BASE_URL,
    temperature=0,
)


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

# ============================================================
# CHROMA
# ============================================================

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
)

# ============================================================
# TEXT SPLITTER
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", ". ", " ", ""],
)

# ============================================================
# LOAD DOCLAYOUT-YOLO
# ============================================================

def load_layout_model():
    print(f"[DocLayout-YOLO] Downloading/loading {DOCLAYOUT_FILENAME}")

    weights_path = hf_hub_download(
        repo_id=DOCLAYOUT_REPO_ID,
        filename=DOCLAYOUT_FILENAME,
    )

    model = YOLOv10(weights_path)
    print("[DocLayout-YOLO] Model loaded.")
    return model


layout_model = load_layout_model()

# ============================================================
# RENDER PDF PAGE
# ============================================================

def render_page(page, scale=2.0):
    matrix = pymupdf.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return image

# ============================================================
# DETECTION
# ============================================================

def _parse_result(image, result):
    """Turn one page's raw YOLO result into our detection dict format."""
    names = result.names  # class_id -> label string
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        raw_label = names[class_id]

        if raw_label not in VISUAL_LABELS:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])

        x1 = max(0, min(x1, image.width))
        y1 = max(0, min(y1, image.height))
        x2 = max(0, min(x2, image.width))
        y2 = max(0, min(y2, image.height))

        if (x2 - x1) < MIN_BOX_SIDE or (y2 - y1) < MIN_BOX_SIDE:
            continue

        # Keep the same label vocabulary the rest of the pipeline expects
        label = "Picture" if raw_label == "figure" else "Table"

        detections.append({
            "label": label,
            "class_id": class_id,
            "confidence": score,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })

    return detections


def detect_visuals_batch(images):
    """
    Run DocLayout-YOLO across many page images at once, in chunks of
    YOLO_BATCH_SIZE. Preprocessing (letterbox resize, normalization) and
    postprocessing (grid decode, NMS) are handled internally by the
    ultralytics-based SDK.

    Returns a list of detection-lists, one per input image, in the same
    order as `images`.
    """
    all_detections = []

    for batch_start in range(0, len(images), YOLO_BATCH_SIZE):
        batch = images[batch_start: batch_start + YOLO_BATCH_SIZE]

        results = layout_model.predict(
            batch,
            imgsz=DOCLAYOUT_IMGSZ,
            conf=CONFIDENCE_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )

        for image, result in zip(batch, results):
            all_detections.append(_parse_result(image, result))

        print(
            f"[DocLayout-YOLO] processed pages "
            f"{batch_start + 1}-{batch_start + len(batch)} of {len(images)}"
        )

    total = sum(len(d) for d in all_detections)
    print(f"[DocLayout-YOLO] {total} visual regions across {len(images)} pages")
    return all_detections

# ============================================================
# CROP VISUAL
# ============================================================

def crop_visual(image, bbox, padding=20):
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)
    return image.crop((x1, y1, x2, y2))


def save_visual(image, source, page_number, visual_id):
    """Persist the cropped figure/table to disk and return its path."""
    doc_stem = Path(source).stem
    out_dir = Path(VISUALS_DIR) / doc_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    file_path = out_dir / f"p{page_number}_{visual_id}.png"
    image.save(file_path, format="PNG")
    return str(file_path)

# ============================================================
# IMAGE -> BASE64
# ============================================================

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# ============================================================
# GEMINI VLM
# ============================================================

def describe_visual(image, source, page_number, visual_number, element_type):
    image_base64 = image_to_base64(image)

    prompt = f"""
You are analyzing a visual element from a research paper.

Document: {source}
Page: {page_number}
Visual: {visual_number}
Element type: {element_type}

Create a detailed factual description of this visual for a research-paper RAG system.
Describe ONLY information visible in the image.

If it is a chart, describe: chart type, title, x-axis, y-axis, units, legend,
variables, categories, important values, trends, comparisons, highest/lowest values.

If it is a diagram, describe: components, inputs, outputs, connections, arrows,
processing stages, architecture, relationships.

If it is a table, describe: column names, row names, important values, units,
comparisons, best/worst results.

Do NOT guess or invent information. If something cannot be read, say:
"Not legible in the provided image."

Return only the description.
"""

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        ]
    )

    try:
        response = llm.invoke([message])
        return response.content.strip()
    except Exception as e:
        print(f"[VLM ERROR] {source} p.{page_number}: {e}")
        return None

# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text_documents(pdf, source):
    documents = []

    for page_index in range(len(pdf)):
        page = pdf[page_index]
        page_number = page_index + 1
        text = page.get_text("text").strip()

        if not text:
            continue

        chunks = text_splitter.split_text(text)

        for chunk_index, chunk in enumerate(chunks):
            chunk_id = f"{source}_p{page_number}_chunk{chunk_index}"
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "page": page_number,
                        "content_type": "text",
                        "chunk_id": chunk_id,
                        "citation": f"[{source}, p. {page_number}]",
                    },
                )
            )

    return documents

# ============================================================
# VISUAL EXTRACTION
# ============================================================

def extract_visual_documents(pdf, source):
    # --- Step 1: render every page up front ---
    page_images = [render_page(pdf[i], scale=2.0) for i in range(len(pdf))]
    print(f"[RENDER] {len(page_images)} pages rendered for {source}")

    # --- Step 2: batched YOLO detection across all pages at once ---
    detections_per_page = detect_visuals_batch(page_images)

    # --- Step 3: crop + save every detected region, building a flat job list ---
    jobs = []
    visual_number = 0

    for page_index, detections in enumerate(detections_per_page):
        page_number = page_index + 1
        page_image = page_images[page_index]

        for detection in detections:
            visual_number += 1
            label = detection["label"]
            bbox = detection["bbox"]

            cropped = crop_visual(page_image, bbox, padding=20)

            if cropped.width < 100 or cropped.height < 100:
                print(f"[SKIP] p.{page_number} visual too small.")
                continue

            content_type = "figure" if label == "Picture" else "table"
            visual_id = f"{content_type}_{visual_number}"

            # Save the crop to disk regardless of whether the VLM call
            # succeeds later, so nothing is lost if Gemini errors out.
            image_path = save_visual(cropped, source, page_number, visual_id)

            jobs.append({
                "cropped": cropped,
                "page_number": page_number,
                "visual_number": visual_number,
                "label": label,
                "content_type": content_type,
                "visual_id": visual_id,
                "image_path": image_path,
            })

    print(f"[VISUAL] {len(jobs)} crops queued for description")

    # --- Step 4: describe all crops concurrently (network-bound VLM calls) ---
    def _describe(job):
        description = describe_visual(
            job["cropped"], source, job["page_number"], job["visual_number"], job["label"]
        )
        return job, description

    documents = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=VLM_MAX_WORKERS) as executor:
        for job, description in executor.map(_describe, jobs):
            if not description:
                print(f"[SKIP] No description for {job['visual_id']}, image saved at {job['image_path']}")
                continue

            content = f"""
[RESEARCH PAPER {job['content_type'].upper()}]

Document: {source}
Page: {job['page_number']}
Visual ID: {job['visual_id']}
Element type: {job['label']}

Description:

{description}
""".strip()

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "page": job["page_number"],
                        "content_type": job["content_type"],
                        "element_type": job["label"],
                        "visual_id": job["visual_id"],
                        "image_path": job["image_path"],
                        "citation": f"[{job['content_type'].capitalize()}: {source}, p. {job['page_number']}]",
                    },
                )
            )

    return documents

# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(pdf_path):
    source = os.path.basename(pdf_path)

    print()
    print("=" * 70)
    print(f"PROCESSING {source}")
    print("=" * 70)

    pdf = pymupdf.open(pdf_path)

    text_documents = extract_text_documents(pdf, source)
    print(f"[TEXT] {len(text_documents)} chunks")

    visual_documents = extract_visual_documents(pdf, source)
    print(f"[VISUAL] {len(visual_documents)} visual descriptions")

    pdf.close()

    return text_documents + visual_documents

# ============================================================
# ADD TO CHROMA
# ============================================================

def add_to_chroma(documents):
    if not documents:
        print("[CHROMA] No documents.")
        return

    ids = []
    for document in documents:
        metadata = document.metadata
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "unknown")
        content_type = metadata.get("content_type", "text")

        if content_type in ("figure", "table"):
            visual_id = metadata.get("visual_id", str(uuid.uuid4()))
            doc_id = f"{source}_p{page}_{visual_id}"
        else:
            doc_id = metadata.get("chunk_id", str(uuid.uuid4()))

        ids.append(doc_id)

    print(f"[CHROMA] Adding {len(documents)} documents...")
    vectorstore.add_documents(documents=documents, ids=ids)
    print("[CHROMA] Done.")

# ============================================================
# MAIN
# ============================================================

def main():
    data_path = Path(DATA_DIR)

    if not data_path.exists():
        raise RuntimeError(f"Data directory not found: {DATA_DIR}")

    pdf_files = list(data_path.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    all_documents = []

    for pdf_file in pdf_files:
        documents = process_pdf(str(pdf_file))
        all_documents.extend(documents)

    text_count = sum(1 for doc in all_documents if doc.metadata.get("content_type") == "text")
    figure_count = sum(1 for doc in all_documents if doc.metadata.get("content_type") == "figure")
    table_count = sum(1 for doc in all_documents if doc.metadata.get("content_type") == "table")

    print()
    print("=" * 70)
    print("INGESTION SUMMARY")
    print("=" * 70)
    print(f"Text chunks : {text_count}")
    print(f"Figures     : {figure_count}")
    print(f"Tables      : {table_count}")
    print(f"Total       : {len(all_documents)}")

    add_to_chroma(all_documents)

    print()
    print("Ingestion completed.")


if __name__ == "__main__":
    main()
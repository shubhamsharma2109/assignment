import os
import re
import base64
import mimetypes

from dotenv import load_dotenv
from openai import OpenAI

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import Chroma

from langchain_community.retrievers import (
    BM25Retriever
)

from langchain_core.documents import Document


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()


GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY is missing from .env"
    )


# ------------------------------------------------------------
# Chroma
# ------------------------------------------------------------

CHROMA_DIR = "chroma_db"

COLLECTION_NAME = "research_papers"


# ------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# ------------------------------------------------------------
# Gemini
# ------------------------------------------------------------

GEMINI_MODEL = "gemini-3.7-flash"

GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/"
    "v1beta/openai/"
)


# ------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------

TOP_K = 5

DENSE_WEIGHT = 0.7

BM25_WEIGHT = 0.3


# ------------------------------------------------------------
# Image settings
# ------------------------------------------------------------

MAX_IMAGES_PER_QUERY = 3


# ============================================================
# GEMINI — OPENAI COMPATIBLE CLIENT
# ============================================================

client = OpenAI(

    api_key=GEMINI_API_KEY,

    base_url=GEMINI_BASE_URL

)


# ============================================================
# EMBEDDINGS
# ============================================================

print(
    "[EMBEDDINGS] Loading model..."
)

embeddings = HuggingFaceEmbeddings(

    model_name=EMBEDDING_MODEL,

    model_kwargs={
        "device": "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True
    }

)

print(
    "[EMBEDDINGS] Model loaded."
)


# ============================================================
# CHROMA
# ============================================================

print(
    "[CHROMA] Loading vector database..."
)

vectorstore = Chroma(

    collection_name=COLLECTION_NAME,

    persist_directory=CHROMA_DIR,

    embedding_function=embeddings

)

print(
    "[CHROMA] Collection:",
    COLLECTION_NAME
)


# ============================================================
# LOAD DOCUMENTS FROM CHROMA
# ============================================================

def load_documents_from_chroma():

    data = vectorstore.get(

        include=[
            "documents",
            "metadatas"
        ]

    )

    documents = []

    for text, metadata in zip(

        data["documents"],

        data["metadatas"]

    ):

        if not text:
            continue

        documents.append(

            Document(

                page_content=text,

                metadata=metadata or {}

            )

        )

    return documents


documents = (
    load_documents_from_chroma()
)


print(
    f"[CHROMA] Loaded "
    f"{len(documents)} documents."
)


# ============================================================
# BM25 RETRIEVER
# ============================================================

if documents:

    bm25_retriever = (
        BM25Retriever.from_documents(
            documents
        )
    )

    bm25_retriever.k = TOP_K

else:

    bm25_retriever = None


# ============================================================
# DENSE RETRIEVER
# ============================================================

dense_retriever = (

    vectorstore.as_retriever(

        search_type="similarity",

        search_kwargs={
            "k": TOP_K
        }

    )

)


# ============================================================
# IMAGE ENCODING
# ============================================================

def encode_image(
    image_path
):

    """
    Convert an image file to base64.

    This follows the OpenAI-compatible
    multimodal API format used by Gemini.
    """

    if not image_path:

        return None


    if not os.path.exists(
        image_path
    ):

        print(
            "[IMAGE] File not found:",
            image_path
        )

        return None


    try:

        with open(
            image_path,
            "rb"
        ) as image_file:

            return base64.b64encode(
                image_file.read()
            ).decode(
                "utf-8"
            )

    except Exception as e:

        print(
            "[IMAGE] Error reading:",
            image_path,
            e
        )

        return None


# ============================================================
# IMAGE -> DATA URL
# ============================================================

def image_to_data_url(
    image_path
):

    encoded = encode_image(
        image_path
    )

    if not encoded:

        return None


    mime_type, _ = (
        mimetypes.guess_type(
            image_path
        )
    )


    if not mime_type:

        mime_type = "image/png"


    return (
        f"data:{mime_type};base64,"
        f"{encoded}"
    )


# ============================================================
# DOCUMENT TYPE
# ============================================================

def document_type(
    doc
):

    content_type = (
        doc.metadata.get(
            "content_type",
            "text"
        )
    )


    if content_type == "figure":

        return "FIGURE"


    if content_type == "table":

        return "TABLE"


    return "TEXT"


# ============================================================
# DOCUMENT KEY
# ============================================================

def document_key(
    doc
):

    metadata = doc.metadata


    source = metadata.get(
        "source",
        ""
    )

    page = metadata.get(
        "page",
        ""
    )

    content_type = metadata.get(
        "content_type",
        "text"
    )

    chunk_id = metadata.get(
        "chunk_id",
        ""
    )

    visual_id = metadata.get(
        "visual_id",
        ""
    )


    return (
        f"{source}|"
        f"{page}|"
        f"{content_type}|"
        f"{chunk_id}|"
        f"{visual_id}"
    )


# ============================================================
# HYBRID RANK FUSION
# ============================================================

def reciprocal_rank_fusion(

    dense_docs,

    bm25_docs,

    rrf_k=60

):

    scores = {}

    doc_map = {}


    # --------------------------------------------------------
    # Dense results
    # --------------------------------------------------------

    for rank, doc in enumerate(

        dense_docs,

        start=1

    ):

        key = document_key(
            doc
        )


        score = (

            DENSE_WEIGHT

            *

            (
                1 /
                (rrf_k + rank)
            )

        )


        scores[key] = (

            scores.get(
                key,
                0
            )

            +

            score

        )


        doc_map[key] = doc


    # --------------------------------------------------------
    # BM25 results
    # --------------------------------------------------------

    for rank, doc in enumerate(

        bm25_docs,

        start=1

    ):

        key = document_key(
            doc
        )


        score = (

            BM25_WEIGHT

            *

            (
                1 /
                (rrf_k + rank)
            )

        )


        scores[key] = (

            scores.get(
                key,
                0
            )

            +

            score

        )


        doc_map[key] = doc


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    ranked = sorted(

        scores.items(),

        key=lambda x: x[1],

        reverse=True

    )


    return [

        doc_map[key]

        for key, _ in ranked

    ][:TOP_K]


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    question
):

    print()
    print(
        "[RETRIEVAL]",
        question
    )


    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------

    dense_docs = (

        dense_retriever.invoke(
            question
        )

    )


    print(
        f"[DENSE] "
        f"{len(dense_docs)} results"
    )


    # --------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------

    if bm25_retriever:

        bm25_docs = (

            bm25_retriever.invoke(
                question
            )

        )

    else:

        bm25_docs = []


    print(
        f"[BM25] "
        f"{len(bm25_docs)} results"
    )


    # --------------------------------------------------------
    # Hybrid fusion
    # --------------------------------------------------------

    docs = reciprocal_rank_fusion(

        dense_docs,

        bm25_docs

    )


    print(
        f"[HYBRID] "
        f"{len(docs)} results"
    )


    return docs


# ============================================================
# FORMAT DOCUMENT
# ============================================================

def format_document(

    doc,

    index

):

    metadata = doc.metadata


    source = metadata.get(

        "source",

        "Unknown"

    )


    page = metadata.get(

        "page",

        "Unknown"

    )


    content_type = document_type(
        doc
    )


    citation = metadata.get(

        "citation",

        "N/A"

    )


    visual_id = metadata.get(

        "visual_id",

        ""

    )


    chunk_id = metadata.get(

        "chunk_id",

        ""

    )


    # --------------------------------------------------------
    # Visual
    # --------------------------------------------------------

    if content_type in (

        "FIGURE",

        "TABLE"

    ):

        location = (

            f"{content_type} "

            f"{visual_id}, "

            f"{source}, "

            f"page {page}"

        )


    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    else:

        location = (

            f"{source}, "

            f"page {page}"

        )


        if chunk_id:

            location += (

                f", chunk {chunk_id}"

            )


    return f"""
[S{index}]
Content type: {content_type}
Source: {source}
Location: {location}
Citation metadata: {citation}

Content:
{doc.page_content}
""".strip()


# ============================================================
# FORMAT CONTEXT
# ============================================================

def format_context(
    docs
):

    context_parts = []


    for i, doc in enumerate(

        docs,

        start=1

    ):

        context_parts.append(

            format_document(

                doc,

                i

            )

        )


    return "\n\n".join(
        context_parts
    )


# ============================================================
# BUILD MULTIMODAL MESSAGE
# ============================================================

def build_message_content(

    question,

    docs

):

    content = []


    # --------------------------------------------------------
    # Text context
    # --------------------------------------------------------

    context = format_context(
        docs
    )


    text = f"""
Retrieved research-paper context:

============================================================

{context}

============================================================

Question:

{question}

============================================================

Instructions:

Answer ONLY using the retrieved research-paper context
and the supplied figure/table images.

Do not use outside knowledge.

If a retrieved item is a figure or table, inspect its
actual image before answering.

Every factual claim must contain a citation such as [S1].

If the answer cannot be determined from the supplied
context and images, respond exactly:

"I don't have enough information in the provided
research papers."
"""


    content.append({

        "type": "text",

        "text": text

    })


    # --------------------------------------------------------
    # Add retrieved images
    # --------------------------------------------------------

    image_count = 0


    for i, doc in enumerate(

        docs,

        start=1

    ):

        if image_count >= (
            MAX_IMAGES_PER_QUERY
        ):

            break


        content_type = doc.metadata.get(

            "content_type",

            "text"

        )


        # Only visual documents
        # contain image_path

        if content_type not in (

            "figure",

            "table"

        ):

            continue


        image_path = doc.metadata.get(

            "image_path"

        )


        if not image_path:

            continue


        data_url = image_to_data_url(

            image_path

        )


        if not data_url:

            continue


        image_count += 1


        # ----------------------------------------------------
        # Image identifier
        # ----------------------------------------------------

        content.append({

            "type": "text",

            "text": (

                f"\n[S{i}] "
                f"This is the actual "
                f"{content_type} image "
                f"corresponding to citation [S{i}]. "
                f"Inspect this image when answering "
                f"the question.\n"

            )

        })


        content.append({

            "type": "image_url",

            "image_url": {

                "url": data_url

            }

        })


    print(

        f"[IMAGE] "
        f"{image_count} visual(s) "
        f"sent to Gemini."

    )


    return content


# ============================================================
# CITATION VALIDATION
# ============================================================

def validate_citations(

    answer,

    number_of_sources

):

    citations = re.findall(

        r"\[S(\d+)\]",

        answer

    )


    # --------------------------------------------------------
    # At least one citation required
    # --------------------------------------------------------

    if not citations:

        return False


    valid_ids = {

        str(i)

        for i in range(

            1,

            number_of_sources + 1

        )

    }


    # --------------------------------------------------------
    # Every citation must exist
    # --------------------------------------------------------

    for citation in citations:

        if citation not in valid_ids:

            return False


    return True


# ============================================================
# ANSWER
# ============================================================

def ask(
    question
):

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    docs = retrieve(
        question
    )


    if not docs:

        return (

            "I don't have enough information "
            "in the provided research papers."

        )


    # --------------------------------------------------------
    # Build multimodal message
    # --------------------------------------------------------

    message_content = (

        build_message_content(

            question,

            docs

        )

    )


    # --------------------------------------------------------
    # System prompt
    # --------------------------------------------------------

    system_prompt = """
You are a research-paper question answering assistant.

SCOPE:

This system is ONLY for answering questions using the
indexed research papers.

GROUNDING:

Use ONLY the supplied context and supplied images.

Never use outside knowledge.

Never guess.

Never invent facts.

Never invent citations.

Never fabricate figure/table values.

If information is not available in the supplied context
or visible in the supplied images, do not infer it.

CITATIONS:

Each retrieved source has an ID:

[S1]
[S2]
[S3]

etc.

Every factual claim must be supported by one or more
of these sources.

For example:

The Transformer uses self-attention mechanisms [S1].

For figure/table claims, cite the source corresponding
to the supplied image.

Only use citation IDs that actually exist.

If the question cannot be answered from the supplied
research papers, respond exactly:

"I don't have enough information in the provided
research papers."

Do not add an explanation to that fallback response.

Keep answers concise and technically accurate.
"""


    # --------------------------------------------------------
    # OpenAI-compatible Gemini call
    # --------------------------------------------------------

    print(
        "[GEMINI] Generating answer..."
    )


    response = client.chat.completions.create(

        model=GEMINI_MODEL,

        messages=[

            {

                "role": "system",

                "content": system_prompt

            },

            {

                "role": "user",

                "content": message_content

            }

        ]

    )


    answer = (

        response
        .choices[0]
        .message
        .content
        .strip()

    )


    # --------------------------------------------------------
    # Gemini fallback
    # --------------------------------------------------------

    if (

        "I don't have enough information"

        in answer

    ):

        return answer


    # --------------------------------------------------------
    # Citation validation
    # --------------------------------------------------------

    if not validate_citations(

        answer,

        len(docs)

    ):

        print(

            "[WARNING] "
            "Citation validation failed."

        )


        return (

            "I don't have enough information "
            "in the provided research papers."

        )


    return answer


# ============================================================
# DEBUG RETRIEVAL
# ============================================================

def debug_retrieval(
    question
):

    docs = retrieve(
        question
    )


    print()
    print(
        "=" * 70
    )

    print(
        "RETRIEVED SOURCES"
    )

    print(
        "=" * 70
    )


    for i, doc in enumerate(

        docs,

        start=1

    ):

        metadata = doc.metadata


        print()

        print(
            f"[S{i}] "
            f"{document_type(doc)}"
        )


        print(
            "Source:",
            metadata.get(
                "source"
            )
        )


        print(
            "Page:",
            metadata.get(
                "page"
            )
        )


        print(
            "Citation:",
            metadata.get(
                "citation"
            )
        )


        print(
            "Visual ID:",
            metadata.get(
                "visual_id"
            )
        )


        print(
            "Image:",
            metadata.get(
                "image_path"
            )
        )


        print(
            "-" * 60
        )


        print(
            doc.page_content[:700]
        )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print()
    print(
        "=" * 70
    )

    print(
        "RESEARCH PAPER MULTIMODAL RAG"
    )

    print(
        "LangChain + Chroma + MiniLM + BM25 + Gemini"
    )

    print(
        "Text + Figure + Table Retrieval"
    )

    print(
        "=" * 70
    )

    print()

    print(
        "Scope: Research-paper based queries only."
    )

    print(
        "Type 'exit' to quit."
    )


    while True:

        question = input(

            "\nQuestion: "

        ).strip()


        if question.lower() == "exit":

            break


        if not question:

            continue


        answer = ask(
            question
        )


        print()
        print(
            "Answer:"
        )

        print(
            answer
        )

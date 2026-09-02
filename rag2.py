import os
import re
import traceback

from dotenv import load_dotenv
from openai import OpenAI

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


# ============================================================
# CONFIGURATION
# ============================================================

print("\n" + "=" * 80)
print("STARTING MULTIMODAL RAG DEBUG")
print("=" * 80)

print("[DEBUG] Loading .env...")
load_dotenv()
print("[DEBUG] .env loaded")


# ------------------------------------------------------------
# Chroma
# ------------------------------------------------------------

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "research_papers"

print("[CONFIG] CHROMA_DIR       =", CHROMA_DIR)
print("[CONFIG] COLLECTION_NAME  =", COLLECTION_NAME)


# ------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

print("[CONFIG] EMBEDDING_MODEL  =", EMBEDDING_MODEL)


# ------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------

TOP_K = 5

DENSE_WEIGHT = 0.7
BM25_WEIGHT = 0.3

print("[CONFIG] TOP_K            =", TOP_K)
print("[CONFIG] DENSE_WEIGHT     =", DENSE_WEIGHT)
print("[CONFIG] BM25_WEIGHT      =", BM25_WEIGHT)


# ============================================================
# vLLM OPENAI-COMPATIBLE CLIENT
# ============================================================

VLLM_BASE_URL = os.getenv(
    "VLLM_BASE_URL",
    "http://127.0.0.1:8000/v1"
)

VLLM_MODEL = os.getenv(
    "VLLM_MODEL",
    "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
)

VLLM_API_KEY = os.getenv(
    "VLLM_API_KEY",
    "EMPTY"
)

print("\n" + "=" * 80)
print("vLLM CONFIGURATION")
print("=" * 80)

print("[VLLM] BASE URL :", VLLM_BASE_URL)
print("[VLLM] MODEL    :", VLLM_MODEL)
print("[VLLM] API KEY  :", VLLM_API_KEY)

print("[VLLM] Creating OpenAI client...")

client = OpenAI(
    api_key=VLLM_API_KEY,
    base_url=VLLM_BASE_URL,
)

print("[VLLM] OpenAI client created")


# ============================================================
# EMBEDDINGS
# ============================================================

print("\n" + "=" * 80)
print("LOADING EMBEDDINGS")
print("=" * 80)

try:

    print("[EMBEDDINGS] Loading model...")

    embeddings = HuggingFaceEmbeddings(

        model_name=EMBEDDING_MODEL,

        model_kwargs={
            "device": "cpu"
        },

        encode_kwargs={
            "normalize_embeddings": True
        }

    )

    print("[EMBEDDINGS] Model loaded successfully")

except Exception as e:

    print("\n[ERROR] Failed to load embeddings")
    print("[ERROR]", repr(e))
    traceback.print_exc()
    raise


# ============================================================
# CHROMA
# ============================================================

print("\n" + "=" * 80)
print("LOADING CHROMA")
print("=" * 80)

try:

    print("[CHROMA] Directory:", os.path.abspath(CHROMA_DIR))

    print("[CHROMA] Loading vector database...")

    vectorstore = Chroma(

        collection_name=COLLECTION_NAME,

        persist_directory=CHROMA_DIR,

        embedding_function=embeddings

    )

    print("[CHROMA] Vector database loaded")

    print(
        "[CHROMA] Collection:",
        COLLECTION_NAME
    )

    # --------------------------------------------------------
    # DEBUG: Check collection count
    # --------------------------------------------------------

    try:

        collection_count = vectorstore._collection.count()

        print(
            "[CHROMA] Number of vectors:",
            collection_count
        )

    except Exception as e:

        print(
            "[CHROMA] Could not get collection count:",
            repr(e)
        )

except Exception as e:

    print("\n[ERROR] Failed to load Chroma")
    print("[ERROR]", repr(e))
    traceback.print_exc()
    raise


# ============================================================
# LOAD DOCUMENTS FROM CHROMA
# ============================================================

def load_documents_from_chroma():

    print("\n" + "=" * 80)
    print("LOADING DOCUMENTS FROM CHROMA")
    print("=" * 80)

    print("[CHROMA] Calling vectorstore.get()...")

    data = vectorstore.get(
        include=[
            "documents",
            "metadatas"
        ]
    )

    print(
        "[CHROMA] Raw documents returned:",
        len(data.get("documents", []))
    )

    print(
        "[CHROMA] Raw metadata returned:",
        len(data.get("metadatas", []))
    )

    documents = []

    for index, (text, metadata) in enumerate(
        zip(
            data["documents"],
            data["metadatas"]
        ),
        start=1
    ):

        print()
        print("-" * 80)
        print(f"[CHROMA DOCUMENT {index}]")
        print("-" * 80)

        print("[DEBUG] Metadata:")
        print(metadata)

        if not text:

            print("[DEBUG] Document has no text. SKIPPING")

            continue

        print("[DEBUG] Text length:", len(text))

        print("[DEBUG] FULL TEXT:")
        print(text)

        documents.append(

            Document(

                page_content=text,

                metadata=metadata or {}

            )

        )

    print()
    print(
        "[CHROMA] Total LangChain documents:",
        len(documents)
    )

    return documents


documents = load_documents_from_chroma()


# ============================================================
# BM25
# ============================================================

print("\n" + "=" * 80)
print("CREATING BM25 RETRIEVER")
print("=" * 80)

if documents:

    print(
        "[BM25] Creating BM25 retriever from",
        len(documents),
        "documents"
    )

    bm25_retriever = BM25Retriever.from_documents(
        documents
    )

    bm25_retriever.k = TOP_K

    print(
        "[BM25] BM25 retriever created"
    )

else:

    print(
        "[BM25] No documents available"
    )

    bm25_retriever = None


# ============================================================
# DENSE RETRIEVER
# ============================================================

print("\n" + "=" * 80)
print("CREATING DENSE RETRIEVER")
print("=" * 80)

dense_retriever = vectorstore.as_retriever(

    search_type="similarity",

    search_kwargs={
        "k": TOP_K
    }

)

print("[DENSE] Dense retriever created")


# ============================================================
# DOCUMENT TYPE
# ============================================================

def document_type(doc):

    content_type = doc.metadata.get(
        "content_type",
        "text"
    )

    if content_type == "figure":
        return "FIGURE"

    if content_type == "table":
        return "TABLE"

    return "TEXT"


# ============================================================
# DOCUMENT KEY
# ============================================================

def document_key(doc):

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

    print("\n" + "=" * 80)
    print("RECIPROCAL RANK FUSION")
    print("=" * 80)

    scores = {}
    doc_map = {}

    # --------------------------------------------------------
    # Dense
    # --------------------------------------------------------

    print(
        "[RRF] Processing dense documents:",
        len(dense_docs)
    )

    for rank, doc in enumerate(
        dense_docs,
        start=1
    ):

        key = document_key(doc)

        score = (
            DENSE_WEIGHT
            *
            (
                1 /
                (rrf_k + rank)
            )
        )

        print(
            f"[RRF][DENSE] Rank={rank} "
            f"Score={score:.6f} "
            f"Key={key}"
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
    # BM25
    # --------------------------------------------------------

    print(
        "[RRF] Processing BM25 documents:",
        len(bm25_docs)
    )

    for rank, doc in enumerate(
        bm25_docs,
        start=1
    ):

        key = document_key(doc)

        score = (
            BM25_WEIGHT
            *
            (
                1 /
                (rrf_k + rank)
            )
        )

        print(
            f"[RRF][BM25] Rank={rank} "
            f"Score={score:.6f} "
            f"Key={key}"
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

    print("\n[RRF] FINAL RANKING")

    for rank, (key, score) in enumerate(
        ranked,
        start=1
    ):

        print(
            f"[RRF] {rank}. "
            f"{score:.6f} "
            f"{key}"
        )

    result = [
        doc_map[key]
        for key, _ in ranked
    ][:TOP_K]

    print(
        "[RRF] Returning",
        len(result),
        "documents"
    )

    return result


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(question):

    print("\n\n" + "=" * 80)
    print("RETRIEVAL START")
    print("=" * 80)

    print("[QUESTION]")
    print(question)

    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------

    print("\n[DENSE] Starting dense retrieval...")

    try:

        dense_docs = dense_retriever.invoke(
            question
        )

        print(
            "[DENSE] Retrieved:",
            len(dense_docs)
        )

    except Exception as e:

        print(
            "[DENSE ERROR]",
            repr(e)
        )

        traceback.print_exc()

        dense_docs = []

    # --------------------------------------------------------
    # Print dense documents
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("DENSE RETRIEVAL RESULTS")
    print("-" * 80)

    for i, doc in enumerate(
        dense_docs,
        start=1
    ):

        print()
        print(f"[DENSE S{i}]")
        print("Metadata:")
        print(doc.metadata)

        print("TEXT:")
        print(doc.page_content)

    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    print("\n[BM25] Starting BM25 retrieval...")

    if bm25_retriever:

        try:

            bm25_docs = bm25_retriever.invoke(
                question
            )

            print(
                "[BM25] Retrieved:",
                len(bm25_docs)
            )

        except Exception as e:

            print(
                "[BM25 ERROR]",
                repr(e)
            )

            traceback.print_exc()

            bm25_docs = []

    else:

        print(
            "[BM25] Retriever unavailable"
        )

        bm25_docs = []

    # --------------------------------------------------------
    # Print BM25 documents
    # --------------------------------------------------------

    print("\n" + "-" * 80)
    print("BM25 RETRIEVAL RESULTS")
    print("-" * 80)

    for i, doc in enumerate(
        bm25_docs,
        start=1
    ):

        print()
        print(f"[BM25 S{i}]")
        print("Metadata:")
        print(doc.metadata)

        print("TEXT:")
        print(doc.page_content)

    # --------------------------------------------------------
    # Hybrid
    # --------------------------------------------------------

    print("\n[HYBRID] Starting fusion...")

    docs = reciprocal_rank_fusion(
        dense_docs,
        bm25_docs
    )

    # --------------------------------------------------------
    # Print final documents
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL HYBRID RETRIEVAL")
    print("=" * 80)

    for i, doc in enumerate(
        docs,
        start=1
    ):

        print()
        print("=" * 80)
        print(f"[FINAL S{i}]")
        print("=" * 80)

        print("TYPE:")
        print(document_type(doc))

        print("\nMETADATA:")
        print(doc.metadata)

        print("\nFULL TEXT:")
        print(doc.page_content)

    print("\n[HYBRID] Final documents:", len(docs))

    return docs


# ============================================================
# FORMAT DOCUMENT
# ============================================================

def format_document(doc, index):

    metadata = doc.metadata

    source = metadata.get(
        "source",
        "Unknown"
    )

    page = metadata.get(
        "page",
        "Unknown"
    )

    content_type = document_type(doc)

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

def format_context(docs):

    print("\n" + "=" * 80)
    print("FORMATTING CONTEXT")
    print("=" * 80)

    context_parts = []

    for i, doc in enumerate(
        docs,
        start=1
    ):

        formatted = format_document(
            doc,
            i
        )

        print("\n" + "-" * 80)
        print(f"CONTEXT SOURCE [S{i}]")
        print("-" * 80)

        print(formatted)

        context_parts.append(
            formatted
        )

    context = "\n\n".join(
        context_parts
    )

    print("\n" + "=" * 80)
    print("FINAL CONTEXT SENT TO MODEL")
    print("=" * 80)

    print(context)

    return context


# ============================================================
# BUILD TEXT-ONLY MESSAGE
# ============================================================

def build_message_content(
    question,
    docs
):

    print("\n" + "=" * 80)
    print("BUILDING TEXT-ONLY MESSAGE")
    print("=" * 80)

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

Answer ONLY using the retrieved research-paper context.

Do not use outside knowledge.

Do not guess.

Every factual claim must contain a citation such as [S1].

If the answer cannot be determined from the supplied
research-paper context, respond exactly:

"I don't have enough information in the provided
research papers."
"""

    print("\n[MODEL MESSAGE]")
    print("=" * 80)
    print(text)
    print("=" * 80)

    # IMPORTANT:
    # TEXT ONLY.
    # NO IMAGE PROCESSING.
    # NO BASE64.
    # NO image_url.

    return [
        {
            "type": "text",
            "text": text
        }
    ]


# ============================================================
# CITATION VALIDATION
# ============================================================

def validate_citations(
    answer,
    number_of_sources
):

    print("\n[CITATION] Validating citations...")

    citations = re.findall(
        r"\[S(\d+)\]",
        answer
    )

    print(
        "[CITATION] Found:",
        citations
    )

    if not citations:

        print(
            "[CITATION] FAILED: No citations"
        )

        return False

    valid_ids = {
        str(i)
        for i in range(
            1,
            number_of_sources + 1
        )
    }

    print(
        "[CITATION] Valid IDs:",
        valid_ids
    )

    for citation in citations:

        if citation not in valid_ids:

            print(
                "[CITATION] INVALID:",
                citation
            )

            return False

    print(
        "[CITATION] Validation successful"
    )

    return True


# ============================================================
# ASK
# ============================================================

def ask(question):

    print("\n\n" + "#" * 80)
    print("ASK() START")
    print("#" * 80)

    print("[ASK] Question:")
    print(question)

    # --------------------------------------------------------
    # Retrieve
    # --------------------------------------------------------

    print("\n[ASK] STEP 1: RETRIEVAL")

    docs = retrieve(
        question
    )

    print(
        "[ASK] Retrieved documents:",
        len(docs)
    )

    if not docs:

        print(
            "[ASK] NO DOCUMENTS RETRIEVED"
        )

        return (
            "I don't have enough information "
            "in the provided research papers."
        )

    # --------------------------------------------------------
    # Build message
    # --------------------------------------------------------

    print("\n[ASK] STEP 2: BUILD MESSAGE")

    message_content = build_message_content(
        question,
        docs
    )

    print(
        "[ASK] Message created"
    )

    # --------------------------------------------------------
    # System prompt
    # --------------------------------------------------------

    system_prompt = """
You are a research-paper question answering assistant.

You MUST answer ONLY using the supplied research-paper
context.

Never use outside knowledge.

Never guess.

Never invent facts.

Never invent citations.

Every factual claim must contain a citation such as [S1].

Only use citation IDs that actually exist.

If the question cannot be answered from the supplied
research-paper context, respond exactly:

"I don't have enough information in the provided
research papers."

Do not add an explanation to that fallback response.

Keep answers concise and technically accurate.
"""

    print("\n[ASK] SYSTEM PROMPT:")
    print(system_prompt)

    # --------------------------------------------------------
    # vLLM request
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("SENDING REQUEST TO vLLM")
    print("=" * 80)

    print("[VLLM] URL:")
    print(VLLM_BASE_URL)

    print("[VLLM] Model:")
    print(VLLM_MODEL)

    print("[VLLM] Calling chat.completions.create()...")

    try:

        response = client.chat.completions.create(

            model=VLLM_MODEL,

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

        print("\n[VLLM] RESPONSE RECEIVED")

        print("=" * 80)

        print("[VLLM RAW RESPONSE OBJECT]")
        print(response)

        print("=" * 80)

    except Exception as e:

        print("\n" + "!" * 80)
        print("VLLM REQUEST FAILED")
        print("!" * 80)

        print("[ERROR TYPE]")
        print(type(e))

        print("\n[ERROR]")
        print(repr(e))

        print("\n[TRACEBACK]")
        traceback.print_exc()

        return (
            f"vLLM request failed: {e}"
        )

    # --------------------------------------------------------
    # Extract answer
    # --------------------------------------------------------

    print("\n[ASK] Extracting model answer...")

    try:

        answer = (
            response
            .choices[0]
            .message
            .content
        )

        print(
            "[ASK] Raw content type:",
            type(answer)
        )

        print("\n[ASK] RAW MODEL ANSWER:")
        print(repr(answer))

        if answer is None:

            print(
                "[ERROR] Model returned None content"
            )

            return (
                "vLLM returned an empty response."
            )

        answer = answer.strip()

    except Exception as e:

        print(
            "[ERROR] Could not extract answer:",
            repr(e)
        )

        traceback.print_exc()

        return (
            f"Could not extract vLLM response: {e}"
        )

    # --------------------------------------------------------
    # Empty answer
    # --------------------------------------------------------

    if not answer:

        print(
            "[ERROR] Model returned EMPTY answer"
        )

        return (
            "vLLM returned an empty answer."
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    if (
        "I don't have enough information"
        in answer
    ):

        print(
            "[ASK] Model returned fallback response"
        )

        return answer

    # --------------------------------------------------------
    # Citation validation
    # --------------------------------------------------------

    print(
        "\n[ASK] STEP 3: CITATION VALIDATION"
    )

    if not validate_citations(
        answer,
        len(docs)
    ):

        print(
            "[WARNING] Citation validation failed"
        )

        print(
            "[WARNING] Returning fallback"
        )

        return (
            "I don't have enough information "
            "in the provided research papers."
        )

    print(
        "[ASK] Citation validation passed"
    )

    print("\n" + "#" * 80)
    print("ASK() FINISHED SUCCESSFULLY")
    print("#" * 80)

    return answer


# ============================================================
# DEBUG RETRIEVAL
# ============================================================

def debug_retrieval(question):

    print("\n" + "=" * 80)
    print("DEBUG RETRIEVAL")
    print("=" * 80)

    docs = retrieve(
        question
    )

    print("\n" + "=" * 80)
    print("DEBUG RETRIEVAL COMPLETE")
    print("=" * 80)

    return docs


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 80)
    print("RESEARCH PAPER TEXT-ONLY RAG DEBUGGER")
    print("=" * 80)

    print(
        "LangChain + Chroma + MiniLM + BM25 + vLLM"
    )

    print(
        "FIGURES/TABLES ARE NOT BEING SENT TO THE MODEL"
    )

    print("=" * 80)

    print()
    print(
        "Scope: Research-paper based queries only."
    )

    print(
        "Type 'exit' to quit."
    )

    print()

    while True:

        try:

            question = input(
                "\nQuestion: "
            ).strip()

        except KeyboardInterrupt:

            print(
                "\n\nExiting..."
            )

            break

        except EOFError:

            print(
                "\n\nEOF received. Exiting..."
            )

            break

        if question.lower() == "exit":

            print(
                "\nExiting..."
            )

            break

        if not question:

            print(
                "[DEBUG] Empty question. Try again."
            )

            continue

        try:

            answer = ask(
                question
            )

            print("\n")
            print("=" * 80)
            print("FINAL ANSWER")
            print("=" * 80)

            print(answer)

            print("=" * 80)

        except Exception as e:

            print("\n" + "!" * 80)
            print("UNEXPECTED ERROR IN MAIN LOOP")
            print("!" * 80)

            print(
                "[ERROR]",
                repr(e)
            )

            traceback.print_exc()

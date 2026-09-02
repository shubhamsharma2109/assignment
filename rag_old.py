import os
import re

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain_huggingface import (
    HuggingFaceEmbeddings
)

from langchain_chroma import Chroma

from langchain_community.retrievers import (
    BM25Retriever
)

from langchain_classic.retrievers import (
    EnsembleRetriever
)

from langchain_core.documents import Document

from langchain_core.prompts import (
    ChatPromptTemplate
)


# ==================================================
# Configuration
# ==================================================

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:

    raise RuntimeError(
        "GEMINI_API_KEY is missing."
    )


CHROMA_DIR = "chroma_db"

COLLECTION_NAME = "rag_documents"

# Requested embedding model
EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

GENERATION_MODEL = "gemini-3.6-flash"

TOP_K = 5


# ==================================================
# Embeddings
# ==================================================

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,

    model_kwargs={
        "device": "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True
    }
)


# ==================================================
# Chroma
# ==================================================

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,

    embedding_function=embeddings,

    collection_name=COLLECTION_NAME
)


# ==================================================
# Dense retrieval
# ==================================================

dense_retriever = vectorstore.as_retriever(
    search_type="similarity",

    search_kwargs={
        "k": TOP_K
    }
)


# ==================================================
# Get documents from Chroma
# ==================================================

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

    documents.append(
        Document(
            page_content=text,
            metadata=metadata or {}
        )
    )


# ==================================================
# BM25 retrieval
# ==================================================

bm25_retriever = BM25Retriever.from_documents(
    documents
)

bm25_retriever.k = TOP_K


# ==================================================
# Hybrid retrieval
# ==================================================

hybrid_retriever = EnsembleRetriever(

    retrievers=[
        dense_retriever,
        bm25_retriever
    ],

    # 70% semantic
    # 30% keyword
    weights=[
        0.7,
        0.3
    ]
)


# ==================================================
# Gemini through OpenAI-compatible API
# ==================================================

llm = ChatOpenAI(

    model=GENERATION_MODEL,

    api_key=GEMINI_API_KEY,

    base_url=(
        "https://generativelanguage.googleapis.com/"
        "v1beta/openai/"
    ),

    temperature=0
)


# ==================================================
# Prompt
# ==================================================

prompt = ChatPromptTemplate.from_messages(
    [

        (
            "system",

            """
You are a grounded Retrieval-Augmented
Generation assistant.

Answer the user's question ONLY using
the provided context.

GROUNDING RULES:

1. Do not use outside knowledge.

2. Do not invent facts.

3. Do not invent sources.

4. Every factual statement must be
   supported by the retrieved context.

5. Cite factual claims using [S1],
   [S2], [S3], etc.

6. Only use citation IDs that actually
   exist in the context.

7. Never create fake citations.

8. If the answer cannot be determined
   from the context, respond:

"I don't have enough information in the
provided documents."

9. Do not guess.

10. Do not use your pretrained knowledge
    to fill missing information.

Keep the answer concise.
"""
        ),

        (
            "human",

            """
Retrieved context:

----------------------------

{context}

----------------------------

Question:

{question}

Answer using ONLY the retrieved context.

Include citations such as [S1] and [S2].
"""
        )
    ]
)


# ==================================================
# Format context
# ==================================================

def format_context(docs):

    context_parts = []

    for i, doc in enumerate(
        docs,
        start=1
    ):

        source = doc.metadata.get(
            "source",
            "Unknown"
        )

        page = doc.metadata.get(
            "page"
        )

        chunk_id = doc.metadata.get(
            "chunk_id",
            "unknown"
        )

        if page is not None:

            location = (
                f"page {page + 1}, "
                f"chunk {chunk_id}"
            )

        else:

            location = (
                f"chunk {chunk_id}"
            )

        context_parts.append(
            f"""
[S{i}]
Source: {source}
Location: {location}

{doc.page_content}
"""
        )

    return "\n\n".join(
        context_parts
    )


# ==================================================
# Validate citations
# ==================================================

def validate_citations(
    answer,
    number_of_sources
):

    citations = re.findall(
        r"\[(S\d+)\]",
        answer
    )

    valid_ids = {
        f"S{i}"
        for i in range(
            1,
            number_of_sources + 1
        )
    }

    for citation in citations:

        if citation not in valid_ids:
            return False

    return True


# ==================================================
# RAG
# ==================================================

def ask(question):

    # ------------------------------
    # Retrieve
    # ------------------------------

    docs = hybrid_retriever.invoke(
        question
    )

    if not docs:

        return (
            "I don't have enough information "
            "in the provided documents."
        )

    # ------------------------------
    # Context
    # ------------------------------

    context = format_context(
        docs
    )

    # ------------------------------
    # Prompt
    # ------------------------------

    messages = prompt.format_messages(
        context=context,
        question=question
    )
    print("context used:", messages)
    # ------------------------------
    # Generate
    # ------------------------------

    response = llm.invoke(
        messages
    )

    answer = response.content

    # ------------------------------
    # Validate citations
    # ------------------------------

    if not validate_citations(
        answer,
        len(docs)
    ):

        return (
            "The generated response contained "
            "an invalid citation and was rejected."
        )

    return answer


# ==================================================
# CLI
# ==================================================

if __name__ == "__main__":

    print("=" * 60)
    print("LangChain + Chroma + MiniLM RAG")
    print("=" * 60)

    print("Type 'exit' to quit.")

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

        print("\nAnswer:")
        print(answer)

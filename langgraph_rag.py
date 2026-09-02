from typing import TypedDict, List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

# ============================================================
# Import existing RAG components from rag.py
# ============================================================

from rag import (
    llm,
    hybrid_retriever,
)


# ============================================================
# Configuration
# ============================================================

MAX_RETRIES = 2


# ============================================================
# State
# ============================================================

class RAGState(TypedDict, total=False):

    question: str

    rewritten_query: str

    documents: List[Document]

    retry_count: int

    route: str

    relevant: bool

    answer: str

    verified: bool

    final_answer: str


# ============================================================
# Fallback
# ============================================================

FALLBACK_MESSAGE = (
    "Not found in the research-paper corpus. "
    "The available papers do not provide sufficient "
    "evidence to answer this question."
)


# ============================================================
# Helper
# ============================================================

def format_context(
    documents: List[Document]
) -> str:

    context_parts = []

    for i, doc in enumerate(
        documents,
        start=1
    ):

        source = doc.metadata.get(
            "source",
            "Unknown source"
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


# ============================================================
# NODE 1: Router
# ============================================================

def router_node(
    state: RAGState
):

    question = state["question"]

    router_prompt = ChatPromptTemplate.from_messages(
        [

            (
                "system",

                """
You are the query router for a
research-paper question answering system.

This system is ONLY intended for
research-paper-based questions.

Classify the user's query into exactly ONE:

RESEARCH_PAPER
GENERAL
OUT_OF_SCOPE


RESEARCH_PAPER:
Questions about research papers, their
methods, experiments, datasets, results,
limitations, findings, related work, etc.


GENERAL:
Simple conversation or general questions
that do not require the research-paper corpus.

Examples:
- Hello
- How are you?
- What is 2 + 2?


OUT_OF_SCOPE:
Requests unrelated to this research-paper
question answering system.

Examples:
- Write a game.
- Book a hotel.
- What is today's weather?


Return ONLY:

RESEARCH_PAPER

or

GENERAL

or

OUT_OF_SCOPE
"""
            ),

            (
                "human",
                "{question}"
            ),
        ]
    )

    messages = router_prompt.format_messages(
        question=question
    )

    response = llm.invoke(
        messages
    )

    result = response.content.strip().upper()

    if "RESEARCH_PAPER" in result:

        route = "RESEARCH_PAPER"

    elif "OUT_OF_SCOPE" in result:

        route = "OUT_OF_SCOPE"

    else:

        route = "GENERAL"

    print(
        f"[ROUTER] {route}"
    )

    return {
        "route": route
    }


# ============================================================
# Router Decision
# ============================================================

def router_decision(
    state: RAGState
):

    route = state["route"]

    if route == "RESEARCH_PAPER":

        return "retrieve"

    if route == "GENERAL":

        return "general"

    return "fallback"


# ============================================================
# NODE 2: Retrieval
# ============================================================

def retrieval_node(
    state: RAGState
):

    question = state["question"]

    retry_count = state.get(
        "retry_count",
        0
    )

    query = state.get(
        "rewritten_query",
        question
    )

    print(
        f"\n[RETRIEVAL] {query}"
    )

    documents = hybrid_retriever.invoke(
        query
    )

    print(
        f"[RETRIEVAL] "
        f"{len(documents)} chunks retrieved"
    )

    return {
        "documents": documents,
        "retry_count": retry_count,
    }


# ============================================================
# NODE 3: Relevance Check
# ============================================================

def relevance_node(
    state: RAGState
):

    question = state["question"]

    documents = state.get(
        "documents",
        []
    )

    if not documents:

        return {
            "relevant": False
        }

    context = format_context(
        documents
    )

    relevance_prompt = ChatPromptTemplate.from_messages(
        [

            (
                "system",

                """
You are a strict relevance grader for
a research-paper RAG system.

Determine whether the retrieved chunks
contain enough information to answer the
question.

Return exactly one:

RELEVANT

or

NOT_RELEVANT

Choose RELEVANT only when the retrieved
text provides evidence that can directly
answer the question.

Do not use outside knowledge.
"""
            ),

            (
                "human",

                """
Question:

{question}


Retrieved research-paper chunks:

{context}
"""
            ),
        ]
    )

    messages = relevance_prompt.format_messages(
        question=question,
        context=context
    )

    response = llm.invoke(
        messages
    )

    result = response.content.strip().upper()

    relevant = (
        result == "RELEVANT"
    )

    print(
        f"[RELEVANCE] {result}"
    )

    return {
        "relevant": relevant
    }


# ============================================================
# Relevance Decision
# ============================================================

def relevance_decision(
    state: RAGState
):

    relevant = state.get(
        "relevant",
        False
    )

    retry_count = state.get(
        "retry_count",
        0
    )

    if relevant:

        return "answer"

    if retry_count < MAX_RETRIES:

        return "rewrite"

    return "fallback"


# ============================================================
# NODE 4: Query Rewrite
# ============================================================

def rewrite_node(
    state: RAGState
):

    question = state["question"]

    previous_query = state.get(
        "rewritten_query",
        question
    )

    retry_count = state.get(
        "retry_count",
        0
    )

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [

            (
                "system",

                """
You rewrite search queries for a
research-paper retrieval system.

Create a better search query that is likely
to retrieve passages directly answering
the user's question.

Rules:

- Preserve the original intent.
- Include important technical terms.
- Include relevant methods, models,
  datasets, metrics, or concepts.
- Do not answer the question.
- Do not invent information.
- Return ONLY the rewritten query.
"""
            ),

            (
                "human",

                """
Original question:

{question}


Previous query:

{previous_query}
"""
            ),
        ]
    )

    messages = rewrite_prompt.format_messages(
        question=question,
        previous_query=previous_query
    )

    response = llm.invoke(
        messages
    )

    rewritten_query = (
        response.content.strip()
    )

    print(
        f"[REWRITE] {rewritten_query}"
    )

    return {
        "rewritten_query": rewritten_query,

        "retry_count": retry_count + 1
    }


# ============================================================
# NODE 5: Answer Generation
# ============================================================

def answer_node(
    state: RAGState
):

    question = state["question"]

    documents = state["documents"]

    context = format_context(
        documents
    )

    answer_prompt = ChatPromptTemplate.from_messages(
        [

            (
                "system",

                """
You are a research-paper QA assistant.

Answer ONLY using the supplied
research-paper context.

STRICT RULES:

1. Do not use outside knowledge.

2. Do not fabricate facts.

3. Do not fabricate research findings.

4. Do not invent paper titles, authors,
   datasets, methods, results, or numbers.

5. Every factual claim must have a citation.

6. Citations must use the supplied IDs:

   [S1]
   [S2]
   [S3]

7. Never create citation IDs that do not
   exist in the context.

8. If the context is insufficient,
   do not guess.

9. Clearly distinguish what the paper
   reports from interpretation.

10. Keep the answer academically clear
    and concise.
"""
            ),

            (
                "human",

                """
Question:

{question}


Research-paper context:

{context}


Answer using ONLY the supplied context.

Include citations such as [S1] and [S2].
"""
            ),
        ]
    )

    messages = answer_prompt.format_messages(
        question=question,
        context=context
    )

    response = llm.invoke(
        messages
    )

    answer = response.content.strip()

    print(
        f"[ANSWER]\n{answer}"
    )

    return {
        "answer": answer
    }


# ============================================================
# NODE 6: Self Verification
# ============================================================

def verification_node(
    state: RAGState
):

    question = state["question"]

    answer = state["answer"]

    documents = state["documents"]

    context = format_context(
        documents
    )

    verification_prompt = ChatPromptTemplate.from_messages(
        [

            (
                "system",

                """
You are a strict hallucination checker
for a research-paper RAG system.

Determine whether the generated answer
is completely supported by the retrieved
research-paper context.

Check:

1. Every factual claim.

2. Whether the context supports each claim.

3. Whether citations refer to real sources.

4. Whether the answer contains information
   not present in the context.

5. Whether the answer fabricates research
   findings, methods, datasets, numbers,
   or conclusions.

6. Whether the answer contradicts the
   retrieved context.

Return exactly:

SUPPORTED

or

UNSUPPORTED

If even one important factual claim is
unsupported, return UNSUPPORTED.

Do not use outside knowledge.
"""
            ),

            (
                "human",

                """
Question:

{question}


Retrieved context:

{context}


Generated answer:

{answer}
"""
            ),
        ]
    )

    messages = verification_prompt.format_messages(
        question=question,
        context=context,
        answer=answer
    )

    response = llm.invoke(
        messages
    )

    result = response.content.strip().upper()

    verified = (
        result == "SUPPORTED"
    )

    print(
        f"[VERIFICATION] {result}"
    )

    return {
        "verified": verified
    }


# ============================================================
# Verification Decision
# ============================================================

def verification_decision(
    state: RAGState
):

    if state.get(
        "verified",
        False
    ):

        return "finish"

    return "fallback"


# ============================================================
# NODE 7: General Answer
# ============================================================

def general_node(
    state: RAGState
):

    question = state["question"]

    response = llm.invoke(
        [
            (
                "system",

                """
You are the general conversation component
of a research-paper QA assistant.

Answer the user's simple general question
normally.

Do NOT claim that the answer came from
the research-paper corpus.
"""
            ),

            (
                "human",
                question
            ),
        ]
    )

    return {
        "final_answer":
            response.content.strip()
    }


# ============================================================
# NODE 8: Fallback
# ============================================================

def fallback_node(
    state: RAGState
):

    return {
        "final_answer":
            FALLBACK_MESSAGE
    }


# ============================================================
# NODE 9: Finish
# ============================================================

def finish_node(
    state: RAGState
):

    return {
        "final_answer":
            state["answer"]
    }


# ============================================================
# Build Graph
# ============================================================

builder = StateGraph(
    RAGState
)


# ============================================================
# Nodes
# ============================================================

builder.add_node(
    "router",
    router_node
)

builder.add_node(
    "retrieve",
    retrieval_node
)

builder.add_node(
    "relevance",
    relevance_node
)

builder.add_node(
    "rewrite",
    rewrite_node
)

builder.add_node(
    "answer",
    answer_node
)

builder.add_node(
    "verify",
    verification_node
)

builder.add_node(
    "general",
    general_node
)

builder.add_node(
    "fallback",
    fallback_node
)

builder.add_node(
    "finish",
    finish_node
)


# ============================================================
# Edges
# ============================================================

builder.add_edge(
    START,
    "router"
)


builder.add_conditional_edges(

    "router",

    router_decision,

    {
        "retrieve": "retrieve",
        "general": "general",
        "fallback": "fallback",
    }
)


builder.add_edge(
    "retrieve",
    "relevance"
)


builder.add_conditional_edges(

    "relevance",

    relevance_decision,

    {
        "answer": "answer",
        "rewrite": "rewrite",
        "fallback": "fallback",
    }
)


builder.add_edge(
    "rewrite",
    "retrieve"
)


builder.add_edge(
    "answer",
    "verify"
)


builder.add_conditional_edges(

    "verify",

    verification_decision,

    {
        "finish": "finish",
        "fallback": "fallback",
    }
)


builder.add_edge(
    "general",
    END
)


builder.add_edge(
    "fallback",
    END
)


builder.add_edge(
    "finish",
    END
)


# ============================================================
# Compile
# ============================================================

rag_graph = builder.compile()


# ============================================================
# Public function
# ============================================================

def ask(
    question: str
):

    initial_state = {

        "question": question,

        "retry_count": 0,

        "documents": [],

        "relevant": False,

        "verified": False,

    }

    result = rag_graph.invoke(
        initial_state
    )

    return result["final_answer"]


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print("=" * 65)

    print(
        "Research Paper RAG"
    )

    print(
        "LangGraph + LangChain + Chroma + MiniLM + Gemini"
    )

    print("=" * 65)

    print(
        "\nThis system is designed ONLY for "
        "research-paper-based queries."
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

        print(
            "\nFinal Answer:"
        )

        print(answer)

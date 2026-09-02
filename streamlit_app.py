import os
import requests
import streamlit as st
# ============================================================
# CONFIG
# ============================================================

RAG_API_URL = os.getenv(
    "RAG_API_URL",
    "http://localhost:8080"
)

ASK_URL = f"{RAG_API_URL}/ask"
HEALTH_URL = f"{RAG_API_URL}/health"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Research Paper RAG",
    page_icon="📚",
    layout="wide",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .rag-title {
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .rag-subtitle {
            color: #777;
            margin-bottom: 1.5rem;
        }

        .citation-card {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 10px;
            background-color: rgba(128, 128, 128, 0.05);
        }

        .citation-title {
            font-weight: 700;
            font-size: 1rem;
            margin-bottom: 5px;
        }

        .citation-meta {
            color: #777;
            font-size: 0.85rem;
        }

        .status-ok {
            color: #16a34a;
            font-weight: 600;
        }

        .status-error {
            color: #dc2626;
            font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# API FUNCTIONS
# ============================================================

def check_api():
    try:
        response = requests.get(
            HEALTH_URL,
            timeout=5,
        )

        if response.ok:
            return True, response.json()

        return False, response.text

    except requests.RequestException as e:
        return False, str(e)


def ask_rag(question):
    try:
        response = requests.post(
            ASK_URL,
            json={
                "question": question
            },
            timeout=300,
        )

        response.raise_for_status()

        return response.json()

    except requests.Timeout:
        return {
            "error": "The RAG API timed out."
        }

    except requests.RequestException as e:
        return {
            "error": f"Could not connect to RAG API: {e}"
        }

    except ValueError:
        return {
            "error": "RAG API returned invalid JSON."
        }


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Configuration")

    st.code(
        RAG_API_URL,
        language="text",
    )

    st.divider()

    st.subheader("API Status")

    if st.button(
        "Check API",
        use_container_width=True,
    ):

        ok, result = check_api()

        if ok:
            st.markdown(
                '<p class="status-ok">● API Online</p>',
                unsafe_allow_html=True,
            )
            st.json(result)

        else:
            st.markdown(
                '<p class="status-error">● API Offline</p>',
                unsafe_allow_html=True,
            )
            st.error(result)

    st.divider()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="rag-title">📚 Research Paper Multimodal RAG</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="rag-subtitle">'
    'Ask questions about your research papers, figures, and tables.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    role = message["role"]

    with st.chat_message(role):

        st.markdown(
            message["content"]
        )

        if role == "assistant":

            citations = message.get(
                "citations",
                []
            )

            path_taken = message.get(
                "path_taken",
                {}
            )

            # ------------------------------------------------
            # Citations
            # ------------------------------------------------

            if citations:

                st.divider()

                st.subheader(
                    "📚 Sources"
                )

                for index, citation in enumerate(
                    citations,
                    start=1,
                ):

                    source = citation.get(
                        "source",
                        "Unknown",
                    )

                    page = citation.get(
                        "page",
                        "Unknown",
                    )

                    content_type = citation.get(
                        "content_type",
                        "TEXT",
                    )

                    visual_id = citation.get(
                        "visual_id",
                        None,
                    )

                    image_path = citation.get(
                        "image_path",
                        None,
                    )

                    citation_text = citation.get(
                        "citation_text",
                        "N/A",
                    )

                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-title">
                                [S{index}] {content_type}
                            </div>

                            <div class="citation-meta">
                                <b>Source:</b> {source}<br>
                                <b>Page:</b> {page}<br>
                                <b>Citation:</b> {citation_text}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if visual_id:
                        st.caption(
                            f"Visual: {visual_id}"
                        )

                    if image_path:

                        # The API container path may not exist
                        # inside the Streamlit container.
                        # We therefore show the path rather than
                        # attempting to open it here.
                        st.code(
                            image_path,
                            language="text",
                        )

            # ------------------------------------------------
            # Graph execution information
            # ------------------------------------------------

            if path_taken:

                with st.expander(
                    "🔍 RAG execution details"
                ):
                    st.json(
                        path_taken
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your research papers..."
)


if question:

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # --------------------------------------------------------
    # API request
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching papers and generating answer..."
        ):

            result = ask_rag(
                question
            )

        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        if "error" in result:

            answer = (
                "❌ **RAG API Error**\n\n"
                + result["error"]
            )

            st.error(
                result["error"]
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": [],
                    "path_taken": {},
                }
            )

        # ----------------------------------------------------
        # Successful response
        # ----------------------------------------------------

        else:

            answer = result.get(
                "answer",
                "No answer returned.",
            )

            citations = result.get(
                "citations",
                [],
            )

            path_taken = result.get(
                "path_taken",
                {},
            )

            st.markdown(
                answer
            )

            # -----------------------------------------------
            # Sources
            # -----------------------------------------------

            if citations:

                st.divider()

                st.subheader(
                    "📚 Sources"
                )

                for index, citation in enumerate(
                    citations,
                    start=1,
                ):

                    source = citation.get(
                        "source",
                        "Unknown",
                    )

                    page = citation.get(
                        "page",
                        "Unknown",
                    )

                    content_type = citation.get(
                        "content_type",
                        "TEXT",
                    )

                    visual_id = citation.get(
                        "visual_id",
                        None,
                    )

                    image_path = citation.get(
                        "image_path",
                        None,
                    )

                    citation_text = citation.get(
                        "citation_text",
                        "N/A",
                    )

                    st.markdown(
                        f"""
                        <div class="citation-card">
                            <div class="citation-title">
                                [S{index}] {content_type}
                            </div>

                            <div class="citation-meta">
                                <b>Source:</b> {source}<br>
                                <b>Page:</b> {page}<br>
                                <b>Citation:</b> {citation_text}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    if visual_id:
                        st.caption(
                            f"Visual: {visual_id}"
                        )

                    if image_path:
                        st.code(
                            image_path,
                            language="text",
                        )

            # -----------------------------------------------
            # Graph state
            # -----------------------------------------------

            if path_taken:

                with st.expander(
                    "🔍 RAG execution details"
                ):
                    st.json(
                        path_taken
                    )

            # -----------------------------------------------
            # Save conversation
            # -----------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "path_taken": path_taken,
                }
            )
import importlib.util
from pathlib import Path

import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Gemini + ChromaDB + LangGraph RAG",
    page_icon="🤖",
    layout="wide",
)


# ============================================================
# LOAD rag-test.py
# ============================================================
#
# The filename contains a hyphen, so normal Python import syntax
# cannot be used. We load the existing working file directly.
# ============================================================

RAG_FILE = Path(__file__).with_name("rag-test.py")

GEMINI_AUTH_ERROR_MESSAGE = (
    "Gemini API key/token expired or invalid. "
    "Please create a new Gemini API key and update GEMINI_API_KEY."
)


def is_gemini_auth_error(error):
    error_text = str(error).lower()

    auth_terms = [
        "gemini_api_key",
        "api key",
        "apikey",
        "authentication",
        "authenticate",
        "unauthenticated",
        "permission_denied",
        "permission denied",
        "invalid_argument",
        "invalid api key",
        "expired",
        "quota",
        "resource_exhausted",
        "403",
        "401",
    ]

    return any(
        term in error_text
        for term in auth_terms
    )


@st.cache_resource
def load_rag_module():
    spec = importlib.util.spec_from_file_location(
        "rag_test",
        RAG_FILE,
    )

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {RAG_FILE}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


@st.cache_resource
def initialize_rag():
    rag = load_rag_module()

    # Initialize the same ChromaDB collections used by rag-test.py.
    rag.initialize_database()

    # Build the exact existing LangGraph:
    # START -> classify -> billing/general/hr -> generate -> END
    app = rag.build_graph()

    return rag, app


# ============================================================
# HEADER
# ============================================================

st.title("🤖 Gemini + ChromaDB + LangGraph RAG")
st.caption(
    "Streamlit UI for the existing working rag-test.py pipeline"
)


# ============================================================
# INITIALIZE
# ============================================================

try:
    rag, app = initialize_rag()
except Exception as error:
    if is_gemini_auth_error(error):
        st.error(GEMINI_AUTH_ERROR_MESSAGE)
    else:
        st.error("Unable to initialize the RAG application.")

    st.exception(error)
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.header("RAG Configuration")

    st.write(f"**Chat model:** `{rag.GEMINI_MODEL}`")
    st.write(f"**Embedding model:** `{rag.EMBEDDING_MODEL}`")
    st.write(f"**ChromaDB:** `{rag.CHROMA_PATH}`")

    st.divider()

    st.subheader("Collections")
    st.write(
        f"Billing documents: **{rag.billing_collection.count()}**"
    )
    st.write(
        f"HR documents: **{rag.hr_collection.count()}**"
    )

    st.divider()

    show_details = st.checkbox(
        "Show RAG details",
        value=False,
        help="Show category and retrieved context after the answer.",
    )

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# CHAT STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# DISPLAY PREVIOUS MESSAGES
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and show_details
            and message.get("category")
        ):
            with st.expander("RAG details"):
                st.write(
                    f"**Category:** `{message['category']}`"
                )

                context = message.get("context", "")

                if context:
                    st.text(context)
                else:
                    st.write("No ChromaDB context used.")


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.chat_input(
    "Ask a billing, HR, or general question..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = rag.process_question(
                    app,
                    question,
                )

                answer = result.get(
                    "answer",
                    "No answer was returned.",
                )

                category = result.get(
                    "category",
                    "unknown",
                )

                context = result.get(
                    "context",
                    "",
                )

                st.markdown(answer)

                if show_details:
                    with st.expander("RAG details"):
                        st.write(
                            f"**Category:** `{category}`"
                        )

                        if context:
                            st.text(context)
                        else:
                            st.write(
                                "No ChromaDB context used."
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "category": category,
                        "context": context,
                    }
                )

            except Exception as error:
                if is_gemini_auth_error(error):
                    error_message = GEMINI_AUTH_ERROR_MESSAGE
                else:
                    error_message = (
                        "Sorry, I was unable to process "
                        "your question."
                    )

                st.error(error_message)
                st.exception(error)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "category": "error",
                        "context": "",
                    }
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Architecture: START → classify → "
    "billing / general / hr → generate → END"
)

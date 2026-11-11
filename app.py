import logging
from pathlib import Path

import streamlit as st

from src.rag_pipeline import RAGPipeline


logger = logging.getLogger(__name__)


def friendly_error_message(exc: Exception) -> str:
    """Show a safe message to users without exposing internal error details."""
    status_code = getattr(exc, "code", None)
    message = str(exc).lower()

    if status_code == 429 or any(
        phrase in message
        for phrase in ("quota", "rate limit", "too many requests", "resource_exhausted")
    ):
        return "The AI service has reached its usage limit. Please try again later."

    if status_code in (401, 403) or any(
        phrase in message
        for phrase in ("api key not valid", "unauthenticated", "permission denied")
    ):
        return "The AI service is not configured correctly. Please contact the app administrator."

    if isinstance(exc, (TimeoutError, ConnectionError)) or any(
        phrase in message
        for phrase in ("timed out", "timeout", "connection refused", "connection error")
    ):
        return "The connection timed out or was interrupted. Please try again."

    if any(phrase in message for phrase in ("qdrant", "vector store", "vector database")):
        return "The knowledge base is temporarily unavailable. Please try again later."

    if "empty response" in message:
        return "The AI service returned an empty response. Please try again."

    return "Something went wrong while processing your question. Please try again later."


st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="AI",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background-color: #f4f7fb;
    }

    .brand {
        font-size: 18px;
        font-weight: 700;
        color: #24527a;
        margin-bottom: 20px;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #17324d;
        margin-bottom: 25px;
    }

    .section-title {
        color: #17324d;
        font-size: 24px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .answer-box {
        background-color: #ffffff;
        border: 1px solid #dce5ed;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 25px;
    }

    .source-box {
        background-color: #ffffff;
        border: 1px solid #dce5ed;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .source-name {
        color: #17324d;
        font-size: 17px;
        font-weight: 600;
        margin-bottom: 5px;
    }

    .source-info {
        color: #6f7f8d;
        font-size: 14px;
    }

    .chunk-label {
        color: #24527a;
        font-size: 15px;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .footer {
        text-align: center;
        color: #7a8b99;
        font-size: 14px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #dce5ed;
    }

    div.stButton > button {
        background-color: #24527a;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 24px;
        font-weight: 600;
    }

    div.stButton > button:hover {
        background-color: #1c4263;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    '<div class="brand">Heuristiclabs.ai</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">AI Interview Assistant</div>',
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    return RAGPipeline()


try:
    pipeline = load_pipeline()
except Exception:
    logger.exception("Failed to initialize the RAG pipeline")
    st.error(
        "Could not initialize the chatbot. "
        "Please check the server configuration and try again later."
    )
    st.stop()


if "result" not in st.session_state:
    st.session_state["result"] = None

if "show_source_document" not in st.session_state:
    st.session_state["show_source_document"] = False


question = st.text_input(
    "Enter your question",
    placeholder="Example: What are your strengths?",
)


if st.button("Ask"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    # Clear the previous answer so a failed request never shows stale results.
    st.session_state["result"] = None
    st.session_state["show_source_document"] = False

    try:
        result = pipeline.answer(question)

        st.session_state["result"] = result
        st.session_state["question"] = question
        st.session_state["show_source_document"] = False

    except Exception as exc:
        logger.exception("Failed to answer chatbot question")
        st.error(friendly_error_message(exc))


result = st.session_state.get("result")


if result is not None:

    st.markdown(
        '<div class="section-title">Answer</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="answer-box">
            {result["answer"]}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Performance</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Retrieval",
        f"{result['retrieval_latency']:.3f}s",
    )

    col2.metric(
        "Generation",
        f"{result['generation_latency']:.3f}s",
    )

    col3.metric(
        "Total",
        f"{result['total_latency']:.3f}s",
    )

    col4.metric(
        "Status",
        result["status"],
    )

    st.markdown(
        '<div class="section-title">Source Document</div>',
        unsafe_allow_html=True,
    )

    source_path = Path(
        "knowledge_base/top 100 common interview questions and best answers.pdf"
    )

    if source_path.exists():

        st.markdown(
            f"""
            <div class="source-box">
                <div class="source-name">
                    {source_path.name}
                </div>
                <div class="source-info">
                    Knowledge-base source document used for retrieval
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("View Source Document"):
            st.session_state["show_source_document"] = True

        if st.session_state.get(
            "show_source_document",
            False,
        ):
            st.pdf(
                str(source_path),
                height=700,
            )

    else:
        st.warning(
            "Source document could not be found."
        )

    st.markdown(
        '<div class="section-title">Retrieved Sources & Chunks</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        f"{len(result['retrieved_chunks'])} chunks retrieved for this question"
    )

    for index, chunk in enumerate(
        result["retrieved_chunks"],
        start=1,
    ):
        metadata = chunk.get(
            "metadata",
            {},
        ) or {}

        page = metadata.get(
            "page",
            "unknown",
        )

        distance = chunk.get(
            "distance",
            0,
        )

        chunk_id = metadata.get(
            "chunk_id",
            chunk.get(
                "id",
                "unknown",
            ),
        )

        with st.expander(
            f"Chunk {index} | Rank {index} | Page {page} | Distance {f'{distance:.4f}' if isinstance(distance, (int, float)) else 'N/A'}"
        ):

            st.markdown(
                f"""
                <div class="chunk-label">
                    Chunk ID: {chunk_id}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write(
                chunk.get(
                    "text",
                    "",
                )
            )


st.markdown(
    '<div class="footer">Powered by heuristiclabs.ai</div>',
    unsafe_allow_html=True,
)

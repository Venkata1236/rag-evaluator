"""
streamlit_app.py
----------------
Streamlit frontend for interacting with the RAG pipeline.
Features:
  - Text input for user questions
  - Displays AI answer with source references
  - Shows RAGAS evaluation scores in sidebar
  - Session-based chat history
"""

# Page config — set title, icon, and layout
# Sidebar: evaluation metrics display and session controls

# Main chat area — render conversation history
# On user submit: call FastAPI /ask endpoint and display response

# Display source documents used to generate the answer
# This improves transparency and trust in the AI response

# streamlit_app.py
# RAG Evaluator — Streamlit Dashboard

import os
import streamlit as st

# ── MUST BE FIRST STREAMLIT COMMAND ──────────────────────────
st.set_page_config(page_title="RAG Evaluator", page_icon="📊", layout="wide")

# ── Load API keys ─────────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY", "")
langsmith_key = os.getenv("LANGCHAIN_API_KEY", "")

if not api_key:
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        langsmith_key = st.secrets.get("LANGCHAIN_API_KEY", "")
    except:
        pass

if langsmith_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    os.environ["LANGCHAIN_PROJECT"] = "rag-evaluator"

from core.rag_chain import build_rag_chain
from core.dataset import EVAL_DATASET
from core.evaluator import run_evaluation, compute_metrics, evaluate_answer

# ── Header ────────────────────────────────────────────────────
st.title("📊 RAG Evaluation Dashboard")
st.caption("Evaluate RAG accuracy — correctness, hallucination, relevance — powered by LangSmith")
st.divider()

# ── Session state ─────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None
if "metrics" not in st.session_state:
    st.session_state.metrics = None

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.info(f"📝 Dataset: {len(EVAL_DATASET)} questions")
    st.divider()

    if langsmith_key:
        st.success("✅ LangSmith tracing enabled")
        st.markdown("[View Traces →](https://smith.langchain.com)")
    else:
        st.warning("⚠️ LangSmith key not set\nAdd LANGCHAIN_API_KEY to .env")

    st.divider()

    if st.button("🚀 Run Evaluation", type="primary", use_container_width=True):
        if not api_key:
            st.error("❌ OPENAI_API_KEY not found.")
        else:
            progress = st.progress(0, text="Building RAG chain...")
            try:
                chain, _ = build_rag_chain(api_key)
                progress.progress(20, text="RAG chain ready!")

                results = []
                for i, item in enumerate(EVAL_DATASET):
                    progress.progress(
                        20 + int((i / len(EVAL_DATASET)) * 70),
                        text=f"Evaluating {i+1}/{len(EVAL_DATASET)}..."
                    )
                    actual = chain.invoke(item["question"])
                    result = evaluate_answer(
                        item["question"],
                        item["expected_answer"],
                        actual,
                        api_key
                    )
                    result["category"] = item.get("category", "general")
                    results.append(result)

                progress.progress(100, text="✅ Done!")
                st.session_state.results = results
                st.session_state.metrics = compute_metrics(results)

            except Exception as e:
                st.error(f"❌ Error: {e}")

    if st.button("🗑️ Clear Results", use_container_width=True):
        st.session_state.results = None
        st.session_state.metrics = None
        st.rerun()

# ── Main area ─────────────────────────────────────────────────
if st.session_state.metrics:
    m = st.session_state.metrics

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Pass Rate", f"{m['pass_rate']}%")
    col2.metric("Passed", m['passed'])
    col3.metric("Failed", m['failed'])
    col4.metric("Avg Correctness", f"{m['avg_correctness']}/10")
    col5.metric("Avg Relevance", f"{m['avg_relevance']}/10")

    st.divider()

    # Results
    st.subheader("📋 Detailed Results")
    for r in st.session_state.results:
        verdict = "✅ PASS" if r["verdict"] == "PASS" else "❌ FAIL"
        with st.expander(f"{verdict} — {r['question'][:70]}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Expected:** {r['expected']}")
                st.markdown(f"**Got:** {r['actual']}")
                st.caption(f"Category: {r.get('category', 'general')}")
            with col2:
                st.metric("Correctness", f"{r['correctness']}/10")
                st.metric("Hallucination", f"{r['hallucination']}/10")
                st.metric("Relevance", f"{r['relevance']}/10")
else:
    st.info("👈 Click **Run Evaluation** in the sidebar to start.")
    st.subheader("📝 Test Dataset Preview")
    for i, item in enumerate(EVAL_DATASET, 1):
        st.markdown(f"**{i}.** {item['question']}")
        st.caption(f"Expected: {item['expected_answer']} | Category: {item['category']}")
# 📊 RAG Evaluator

> Evaluate RAG accuracy — correctness, hallucination detection, relevance — powered by LangSmith

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.3.7-green)
![LangSmith](https://img.shields.io/badge/LangSmith-tracing-purple)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5--turbo-orange)
![FAISS](https://img.shields.io/badge/FAISS-vector--store-red)
![Streamlit](https://img.shields.io/badge/Streamlit-1.41.1-red)

---

## 📌 What Is This?

A RAG evaluation system that measures how accurately a RAG pipeline answers questions. Uses LangSmith for tracing every LLM call, and LLM-as-judge to score correctness, hallucination, and relevance for each answer.

---

## 🗺️ Simple Flow

```
hr_policies.txt
        ↓
Build FAISS RAG chain
        ↓
Run 10 test questions from dataset
        ↓
LLM-as-judge scores each answer:
  - Correctness  (0-10)
  - Hallucination (0-10)
  - Relevance    (0-10)
  - Verdict: PASS/FAIL
        ↓
LangSmith traces every call
        ↓
Streamlit dashboard shows results
```

---

## 🏗️ Architecture

```
rag_evaluator/
├── app.py                    ← Terminal evaluation runner
├── streamlit_app.py          ← Evaluation dashboard
├── data/
│   └── hr_policies.txt       ← HR policy document
├── core/
│   ├── tracer.py             ← LangSmith tracing setup
│   ├── rag_chain.py          ← FAISS RAG pipeline
│   ├── dataset.py            ← 10 test Q&A pairs
│   └── evaluator.py          ← LLM-as-judge evaluation
└── requirements.txt
```

---

## 🧠 Key Concepts

| Concept | What It Does |
|---|---|
| **LangSmith Tracing** | See every LLM call — prompt, response, tokens, latency |
| **LLM-as-judge** | Use GPT to evaluate GPT's answers |
| **Correctness** | Does the RAG answer match the expected answer? |
| **Hallucination** | Did the model make up facts not in the document? |
| **Relevance** | Is the answer relevant to the question asked? |
| **Dataset** | Ground truth Q&A pairs used to measure accuracy |

---

## 📊 Evaluation Metrics

| Metric | Range | Meaning |
|---|---|---|
| Correctness | 0-10 | How correct is the answer vs expected |
| Hallucination | 0-10 | 10 = no hallucination, 0 = lots of made-up facts |
| Relevance | 0-10 | How relevant is the answer to the question |
| Pass Rate | % | Percentage of questions that got PASS verdict |

---

## ⚙️ Setup

**Step 1 — Install:**
```bash
pip install -r requirements.txt
```

**Step 2 — Get LangSmith API key:**
1. Go to [smith.langchain.com](https://smith.langchain.com)
2. Sign up free
3. Settings → API Keys → Create Key

**Step 3 — Add to `.env`:**
```
OPENAI_API_KEY=sk-your-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_PROJECT=rag-evaluator
```

**Step 4 — Run:**
```bash
# Streamlit dashboard
python -m streamlit run streamlit_app.py

# Terminal
python app.py
```

---

## 📦 Tech Stack

- **LangSmith** — Tracing, observability
- **LangChain** — RAG pipeline, LCEL chain
- **FAISS** — Vector store
- **OpenAI** — GPT-3.5-turbo for RAG + evaluation
- **Streamlit** — Dashboard UI

---

## 👤 Author

**Venkata Reddy Bommavaram**
- 📧 bommavaramvenkat2003@gmail.com
- 💼 [LinkedIn](https://linkedin.com/in/venkatareddy1203)
- 🐙 [GitHub](https://github.com/venkata1236)
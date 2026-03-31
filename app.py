# app.py
# Terminal entry point — Run RAG evaluation

import os
from dotenv import load_dotenv
from core.tracer import setup_tracing
from core.rag_chain import build_rag_chain
from core.dataset import EVAL_DATASET
from core.evaluator import run_evaluation, compute_metrics

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    print("❌ ERROR: Set OPENAI_API_KEY in .env")
    exit(1)


def main():
    print("\n" + "=" * 60)
    print("       📊 RAG EVALUATOR")
    print("   LangSmith + Precision + Recall + Hallucination")
    print("=" * 60)

    # Setup LangSmith tracing
    setup_tracing("rag-evaluator")

    # Build RAG chain
    print("\n⏳ Building RAG chain...")
    chain, retriever = build_rag_chain(API_KEY)

    # Run evaluation
    print(f"\n⏳ Running evaluation on {len(EVAL_DATASET)} questions...")
    results = run_evaluation(chain, EVAL_DATASET, API_KEY)

    # Compute metrics
    metrics = compute_metrics(results)

    # Display results
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    for r in results:
        verdict = "✅" if r["verdict"] == "PASS" else "❌"
        print(f"\n{verdict} Q: {r['question'][:60]}")
        print(f"   Expected : {r['expected'][:60]}")
        print(f"   Got      : {r['actual'][:60]}")
        print(f"   Scores   : Correctness={r['correctness']} | Hallucination={r['hallucination']} | Relevance={r['relevance']}")

    print("\n" + "=" * 60)
    print("📈 METRICS SUMMARY")
    print("=" * 60)
    for key, val in metrics.items():
        print(f"  {key}: {val}")

    print(f"\n🔗 View full traces: https://smith.langchain.com")


if __name__ == "__main__":
    main()
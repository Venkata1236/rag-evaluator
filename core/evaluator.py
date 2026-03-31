"""
evaluator.py
------------
Evaluates RAG pipeline quality using RAGAS framework.

Metrics computed:
  - faithfulness     : Is the answer grounded in context?
  - answer_relevancy : Is the answer relevant to the question?
  - context_recall   : Does context cover the ground truth?
  - context_precision: Is retrieved context actually useful?
"""

# Initialize RAGAS metrics — customize which metrics to run
# Higher faithfulness = less hallucination
# Higher context_precision = better retriever performance

# Convert results to pandas DataFrame for easy inspection and export
# Save scores to CSV for version tracking across runs


# core/evaluator.py
# Evaluate RAG accuracy — precision, recall, hallucination detection
# Concept: LangSmith evaluators + custom metrics

import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate


def evaluate_answer(
    question: str,
    expected_answer: str,
    actual_answer: str,
    api_key: str
) -> Dict:
    """
    Evaluates a single RAG answer using LLM-as-judge.

    Checks:
    - Correctness  : does actual answer match expected?
    - Hallucination: did the model make up facts?
    - Relevance    : is the answer relevant to the question?

    Args:
        question       : the original question
        expected_answer: ground truth answer
        actual_answer  : RAG system's answer
        api_key        : OpenAI API key

    Returns:
        dict with scores and feedback
    """
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.0,
        openai_api_key=api_key
    )

    prompt = PromptTemplate.from_template("""
You are evaluating a RAG system's answer quality.

Question: {question}
Expected Answer: {expected_answer}
Actual Answer: {actual_answer}

Evaluate on these 3 criteria. Respond in EXACTLY this format:
CORRECTNESS: [0-10] - [one line explanation]
HALLUCINATION: [0-10] - [one line explanation] (0=lots of hallucination, 10=no hallucination)
RELEVANCE: [0-10] - [one line explanation]
VERDICT: [PASS/FAIL]
""")

    response = llm.invoke(prompt.format(
        question=question,
        expected_answer=expected_answer,
        actual_answer=actual_answer
    ))

    return parse_evaluation(response.content, question, expected_answer, actual_answer)


def parse_evaluation(raw: str, question: str, expected: str, actual: str) -> Dict:
    """Parses the LLM evaluation response into a structured dict."""
    result = {
        "question": question,
        "expected": expected,
        "actual": actual,
        "correctness": 0,
        "hallucination": 0,
        "relevance": 0,
        "verdict": "FAIL",
        "raw": raw
    }

    for line in raw.split("\n"):
        line = line.strip()
        if line.startswith("CORRECTNESS:"):
            try:
                result["correctness"] = int(line.split(":")[1].strip().split()[0].split("-")[0].strip())
            except:
                pass
        elif line.startswith("HALLUCINATION:"):
            try:
                result["hallucination"] = int(line.split(":")[1].strip().split()[0].split("-")[0].strip())
            except:
                pass
        elif line.startswith("RELEVANCE:"):
            try:
                result["relevance"] = int(line.split(":")[1].strip().split()[0].split("-")[0].strip())
            except:
                pass
        elif line.startswith("VERDICT:"):
            result["verdict"] = "PASS" if "PASS" in line else "FAIL"

    return result


def run_evaluation(chain, dataset: List[Dict], api_key: str) -> List[Dict]:
    """
    Runs full evaluation on all dataset questions.

    Args:
        chain  : RAG chain to evaluate
        dataset: list of Q&A pairs
        api_key: OpenAI API key

    Returns:
        list of evaluation results
    """
    results = []
    for i, item in enumerate(dataset):
        print(f"  Evaluating {i+1}/{len(dataset)}: {item['question'][:50]}...")
        actual = chain.invoke(item["question"])
        eval_result = evaluate_answer(
            item["question"],
            item["expected_answer"],
            actual,
            api_key
        )
        eval_result["category"] = item.get("category", "general")
        results.append(eval_result)

    return results


def compute_metrics(results: List[Dict]) -> Dict:
    """
    Computes overall metrics from evaluation results.

    Returns:
        dict with avg scores and pass rate
    """
    total = len(results)
    if total == 0:
        return {}

    passed = sum(1 for r in results if r["verdict"] == "PASS")

    return {
        "total_questions": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total * 100, 1),
        "avg_correctness": round(sum(r["correctness"] for r in results) / total, 1),
        "avg_hallucination": round(sum(r["hallucination"] for r in results) / total, 1),
        "avg_relevance": round(sum(r["relevance"] for r in results) / total, 1),
    }
"""
dataset.py
----------
Handles loading and preparing the evaluation dataset.
Each entry contains: question, ground_truth answer, and context.
Used by evaluator.py to run RAGAS metrics.
"""

# Load dataset from local JSON or HuggingFace dataset
# Format expected: [{"question": str, "ground_truth": str, "contexts": list}]

# Convert to HuggingFace Dataset format required by RAGAS


# core/dataset.py
# Test questions and expected answers for RAG evaluation
# Concept: Dataset — ground truth Q&A pairs used to measure RAG accuracy

# Each item has:
# question        → what we ask the RAG system
# expected_answer → what the correct answer should be
# category        → type of question (for grouping results)

EVAL_DATASET = [
    {
        "question": "How many annual leave days do employees get?",
        "expected_answer": "21 days per year",
        "category": "leave"
    },
    {
        "question": "How many sick leave days are employees entitled to?",
        "expected_answer": "10 days of paid sick leave per year",
        "category": "leave"
    },
    {
        "question": "How many days can employees work from home per week?",
        "expected_answer": "3 days per week",
        "category": "wfh"
    },
    {
        "question": "What are the core working hours for WFH?",
        "expected_answer": "10 AM to 4 PM",
        "category": "wfh"
    },
    {
        "question": "How long is maternity leave?",
        "expected_answer": "26 weeks fully paid",
        "category": "leave"
    },
    {
        "question": "When is payroll processed?",
        "expected_answer": "25th of each month",
        "category": "payroll"
    },
    {
        "question": "What is the employee referral bonus amount?",
        "expected_answer": "$2000",
        "category": "compensation"
    },
    {
        "question": "How many days in advance must WFH be approved?",
        "expected_answer": "Must be approved by the direct manager",
        "category": "wfh"
    },
    {
        "question": "When are performance reviews conducted?",
        "expected_answer": "Twice a year in June and December",
        "category": "performance"
    },
    {
        "question": "How many days can employees work remotely from another country?",
        "expected_answer": "Up to 30 days per year",
        "category": "remote"
    },
]
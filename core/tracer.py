# core/tracer.py
# LangSmith tracing setup
# Concept: Tracing — see every LLM call in LangSmith dashboard

import os
from dotenv import load_dotenv

load_dotenv()


def setup_tracing(project_name: str = "rag-evaluator"):
    """
    Enables LangSmith tracing for all LangChain calls.
    Just setting env variables is enough — LangChain detects them automatically.

    Args:
        project_name: name of the project in LangSmith dashboard
    """
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = project_name

    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print("⚠️  LANGCHAIN_API_KEY not set — tracing disabled")
        return False

    print(f"✅ LangSmith tracing enabled → project: '{project_name}'")
    print(f"   View traces at: https://smith.langchain.com")
    return True


def get_run_url(run_id: str) -> str:
    """Returns the LangSmith URL for a specific run."""
    return f"https://smith.langchain.com/runs/{run_id}"
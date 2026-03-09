"""Configuration for PRISM evaluation framework."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)  # Override shell env vars with .env values

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "dataset" / "prism_eval_dataset.jsonl"
RESULTS_DIR = EVAL_DIR / "results"
RAW_DIR = RESULTS_DIR / "raw"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"
LATEX_DIR = RESULTS_DIR / "latex"
CACHE_DIR = RESULTS_DIR / "cache"

# ── API Keys ───────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ── Models ─────────────────────────────────────────────────────────────────
SYSTEM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
JUDGE_MODEL_PRIMARY = "claude-sonnet-4-6"       # Anthropic (primary judge)
JUDGE_MODEL_SECONDARY = "gpt-4.1-mini"          # OpenAI (validation judge)

# ── Variants ───────────────────────────────────────────────────────────────
VARIANTS = [
    "full_system",
    "no_rag",
    "no_personalization",
    "no_internal_eval",
    "no_web_search",
    "no_query_refinement",
    "baseline",
]

VARIANT_DESCRIPTIONS = {
    "full_system": "All 7 agents enabled (baseline system)",
    "no_rag": "Course RAG agent disabled, web search only",
    "no_personalization": "Personalization agent skipped",
    "no_internal_eval": "Internal evaluation/refinement loop skipped",
    "no_web_search": "Web search fallback disabled",
    "no_query_refinement": "Query refinement agent disabled",
    "baseline": "Plain LLM (GPT-4o) with no RAG, no agents",
}

# ── Profiles ───────────────────────────────────────────────────────────────
PROFILES = {
    "undergrad": {
        "degree": "Bachelors",
        "major": "Information Science",
        "student_id": "eval_undergrad_001",
    },
    "masters": {
        "degree": "Masters",
        "major": "Data Science",
        "student_id": "eval_masters_001",
    },
    "phd": {
        "degree": "PhD",
        "major": "Information Science",
        "student_id": "eval_phd_001",
    },
}

# ── Metrics ────────────────────────────────────────────────────────────────
RAG_METRICS = [
    "faithfulness",
    "answer_relevancy",
    "contextual_precision",
    "contextual_recall",
    "correctness",
    "hallucination",
]

AGENT_METRICS = [
    "tool_correctness",
    "task_completion",
    "routing_accuracy",
    "clarification_quality",
    "refusal_correctness",
]

RESPONSE_METRICS = [
    "coherence",
    "readability",
    "personalization_accuracy",
]

SAFETY_METRICS = [
    "toxicity",
    "bias",
]

ALL_METRICS = RAG_METRICS + AGENT_METRICS + RESPONSE_METRICS + SAFETY_METRICS

# Metrics applicable per category
METRICS_BY_CATEGORY = {
    "course_based": RAG_METRICS + ["tool_correctness", "task_completion", "routing_accuracy",
                                    "coherence", "readability", "personalization_accuracy",
                                    "toxicity", "bias"],
    "web_required": RAG_METRICS + ["tool_correctness", "task_completion", "routing_accuracy",
                                    "coherence", "readability", "personalization_accuracy",
                                    "toxicity", "bias"],
    "multi_hop": RAG_METRICS + ["tool_correctness", "task_completion", "routing_accuracy",
                                 "coherence", "readability", "personalization_accuracy",
                                 "toxicity", "bias"],
    "vague": ["clarification_quality", "routing_accuracy", "task_completion", "toxicity", "bias"],
    "out_of_scope": ["refusal_correctness", "routing_accuracy", "task_completion", "toxicity", "bias"],
}

# ── Runner Settings ────────────────────────────────────────────────────────
MAX_WORKERS = 12
BATCH_SIZE = 50
TIMEOUT_PER_QUERY = 180  # seconds
MAX_RETRIES = 2

# All variants use all 3 profiles — personalization node is active in all except
# no_personalization and baseline, so profile affects output for all variants.
FULL_PROFILE_VARIANTS = {"full_system", "no_personalization", "baseline",
                         "no_rag", "no_internal_eval", "no_web_search", "no_query_refinement"}
SINGLE_PROFILE_VARIANTS = set()  # None — all variants get all profiles

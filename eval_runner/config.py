"""Ablation configs and paths for PRISM evaluation runner."""

from pathlib import Path
from typing import List, Dict, Any

# Ablation variants
VARIANTS = [
    "llm_only",
    "retriever_only",
    "no_personalization",
    "no_internal_eval",
    "full_system",
]

# Default profiles to test (align with dataset personalization_targets)
DEFAULT_PROFILES = ["undergrad", "masters", "phd"]

# Course ID -> course name (for PRISM API)
COURSE_ID_TO_NAME: Dict[str, str] = {
    "INFO4100": "INFO 4100-Introduction to Information Sciences",
}

# Profile -> user_context for PRISM
PROFILE_TO_USER_CONTEXT: Dict[str, Dict[str, Any]] = {
    "undergrad": {
        "degree": "Bachelor of Science",
        "major": "Information Science",
        "student_id": "eval_undergrad",
    },
    "masters": {
        "degree": "Master of Science",
        "major": "Information Science",
        "student_id": "eval_masters",
    },
    "phd": {
        "degree": "Doctor of Philosophy",
        "major": "Information Science",
        "student_id": "eval_phd",
    },
}

# Expected tools by category (for tool_correctness)
CATEGORY_EXPECTED_TOOLS = {
    "vague": [],
    "out_of_scope": [],
    "web_required": ["web_search"],
    "course_based": ["vector_retrieval"],
}

# Results subdirs
RAW_DIR = "raw"
TABLES_DIR = "tables"
PLOTS_DIR = "plots"
LATEX_DIR = "latex"
CACHE_DIR = "cache"
JUDGE_CACHE_FILENAME = "judge_cache.jsonl"
SUMMARY_JSON = "summary.json"


def get_course_name(course_id: str) -> str:
    """Resolve course_id to PRISM course name."""
    return COURSE_ID_TO_NAME.get(course_id, course_id)


def ensure_results_dirs(results_dir: Path) -> None:
    """Create raw/, tables/, plots/, latex/, cache/ under results_dir."""
    for d in (RAW_DIR, TABLES_DIR, PLOTS_DIR, LATEX_DIR, CACHE_DIR):
        (results_dir / d).mkdir(parents=True, exist_ok=True)

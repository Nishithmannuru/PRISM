"""
Step 2 — Retrieve candidate chunks (course_based only).

For each course_based question: query the course vector store with top_k=15,
then save candidate_chunks with chunk_id, doc_id, page_or_slide, chunk_text.

Usage (from project root):
  python -m evaluation.retrieve_candidates
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.retriever import CourseRetriever

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

COURSE_NAME = "INFO 4100-Introduction to Information Sciences"
QUESTIONS_PATH = Path(__file__).parent / "questions_INFO4100.json"
CANDIDATES_PATH = Path(__file__).parent / "candidates_INFO4100.json"
TOP_K = 15


def chunk_to_candidate(chunk: dict) -> dict:
    """Convert retriever chunk to evaluation candidate format."""
    page_or_slide = chunk.get("page_number") or chunk.get("timestamp") or None
    return {
        "chunk_id": chunk.get("id"),
        "doc_id": chunk.get("document_name"),
        "page_or_slide": page_or_slide,
        "chunk_text": chunk.get("content", ""),
    }


def main():
    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    course_based = [q for q in questions if q["category"] == "course_based"]
    logger.info(f"Found {len(course_based)} course_based questions.")

    retriever = CourseRetriever()
    results = {}

    for item in course_based:
        eval_id = item["eval_id"]
        question = item["question"]
        logger.info(f"Retrieving top-{TOP_K} for: {eval_id}")
        chunks = retriever.retrieve(
            query=question,
            course_name=COURSE_NAME,
            top_k=TOP_K,
        )
        candidates = [chunk_to_candidate(c) for c in chunks]
        results[eval_id] = {
            "eval_id": eval_id,
            "question": question,
            "course_id": "INFO4100",
            "candidate_chunks": candidates,
        }

    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CANDIDATES_PATH, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Wrote candidates to {CANDIDATES_PATH}")


if __name__ == "__main__":
    main()

"""
Steps 5–7 — Attach expected behavior, personalization targets, write eval dataset as JSONL.

For every question:
- expected_behavior (route, needs_clarification, should_refuse, expected_tools)
- For answerable: personalization_targets (undergrad, masters, phd grade bands)
- For course_based: candidate_chunks, gold_chunks from prior steps
- For course_based + web_required: required_keypoints from keypoints step
- For vague: clarification_requirements (placeholder; can be filled by LLM later)
- For out_of_scope: refusal_requirements (placeholder)

Output: course_eval_dataset.jsonl (one JSON object per line).

Requires: questions_INFO4100.json; optionally candidates, gold_chunks, keypoints.

Usage (from project root):
  python -m evaluation.build_dataset
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
QUESTIONS_PATH = EVAL_DIR / "questions_INFO4100.json"
CANDIDATES_PATH = EVAL_DIR / "candidates_INFO4100.json"
GOLD_CHUNKS_PATH = EVAL_DIR / "gold_chunks_INFO4100.json"
KEYPOINTS_PATH = EVAL_DIR / "keypoints_INFO4100.json"
OUTPUT_PATH = EVAL_DIR / "course_eval_dataset.jsonl"

COURSE_ID = "INFO4100"

EXPECTED_BEHAVIOR = {
    "vague": {
        "expected_route": "clarify",
        "needs_clarification": True,
        "should_refuse": False,
        "expected_tools": [],
    },
    "out_of_scope": {
        "expected_route": "refuse",
        "needs_clarification": False,
        "should_refuse": True,
        "expected_tools": [],
    },
    "web_required": {
        "expected_route": "web",
        "needs_clarification": False,
        "should_refuse": False,
        "expected_tools": ["web_search"],
    },
    "course_based": {
        "expected_route": "course",
        "needs_clarification": False,
        "should_refuse": False,
        "expected_tools": ["vector_retrieval"],
    },
}

PERSONALIZATION_TARGETS = {
    "undergrad": {"target_grade_band": [9, 12]},
    "masters": {"target_grade_band": [12, 15]},
    "phd": {"target_grade_band": [15, 18]},
}


def main():
    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    candidates_data = {}
    if CANDIDATES_PATH.exists():
        with open(CANDIDATES_PATH, "r") as f:
            candidates_data = json.load(f)

    gold_data = {}
    if GOLD_CHUNKS_PATH.exists():
        with open(GOLD_CHUNKS_PATH, "r") as f:
            gold_data = json.load(f)

    keypoints_data = {}
    if KEYPOINTS_PATH.exists():
        with open(KEYPOINTS_PATH, "r") as f:
            keypoints_data = json.load(f)

    records = []

    for item in questions:
        eval_id = item["eval_id"]
        category = item["category"]
        question = item["question"]

        rec = {
            "eval_id": eval_id,
            "course_id": COURSE_ID,
            "category": category,
            "question": question,
            "expected_behavior": EXPECTED_BEHAVIOR[category],
        }

        if category == "vague":
            rec["clarification_requirements"] = {
                "what_missing_information_should_be_requested": "placeholder — define what should be asked to disambiguate",
            }
        elif category == "out_of_scope":
            rec["refusal_requirements"] = {
                "polite_refusal": True,
                "scope_explanation": True,
                "redirect_to_course_topics": True,
            }
        elif category in ("course_based", "web_required"):
            rec["personalization_targets"] = PERSONALIZATION_TARGETS
            kp = keypoints_data.get(eval_id, {})
            rec["required_keypoints"] = kp.get("required_keypoints", [])

        if category == "course_based":
            cand = candidates_data.get(eval_id, {})
            rec["candidate_chunks"] = cand.get("candidate_chunks", [])
            gold = gold_data.get(eval_id, {})
            rec["gold_chunks"] = gold.get("gold_chunk_ids", [])

        records.append(rec)

    with open(OUTPUT_PATH, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info(f"Wrote {len(records)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

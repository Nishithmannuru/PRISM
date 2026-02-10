"""
Step 4 — Extract required keypoints (course_based + web_required).

- course_based: use gold course chunks (from gold_chunks + candidates) to extract 3–7 keypoints.
- web_required: use general domain knowledge to extract 3–7 keypoints.

Each keypoint: factual, verifiable, necessary for a correct answer.

Requires: questions_INFO4100.json, candidates_INFO4100.json, gold_chunks_INFO4100.json
(run retrieve_candidates.py and select_gold_chunks.py first for course_based).

Usage (from project root):
  python -m evaluation.extract_keypoints
"""

import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI
from config.settings import OPENAI_API_KEY, OPENAI_MODEL

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
QUESTIONS_PATH = EVAL_DIR / "questions_INFO4100.json"
CANDIDATES_PATH = EVAL_DIR / "candidates_INFO4100.json"
GOLD_CHUNKS_PATH = EVAL_DIR / "gold_chunks_INFO4100.json"
KEYPOINTS_PATH = EVAL_DIR / "keypoints_INFO4100.json"

SYSTEM_KEYPOINTS = """You are preparing evaluation ground truth.
Extract 3–7 atomic keypoints that a correct answer to the question MUST cover.

Rules:
- Each keypoint must be: factual, verifiable, and necessary for a correct answer.
- Keypoints should be independent of phrasing (suitable for undergrad through PhD).
- No full-sentence answers — concise factual claims only.

Respond with valid JSON only: {"required_keypoints": ["keypoint 1", "keypoint 2", ...]}"""


def main():
    with open(QUESTIONS_PATH, "r") as f:
        questions = json.load(f)

    answerable = [q for q in questions if q["category"] in ("course_based", "web_required")]
    logger.info(f"Extracting keypoints for {len(answerable)} answerable questions.")

    candidates_data = {}
    if CANDIDATES_PATH.exists():
        with open(CANDIDATES_PATH, "r") as f:
            candidates_data = json.load(f)

    gold_data = {}
    if GOLD_CHUNKS_PATH.exists():
        with open(GOLD_CHUNKS_PATH, "r") as f:
            gold_data = json.load(f)

    client = OpenAI(api_key=OPENAI_API_KEY)
    results = {}

    for item in answerable:
        eval_id = item["eval_id"]
        category = item["category"]
        question = item["question"]

        if category == "course_based":
            gold = gold_data.get(eval_id, {})
            gold_ids = set(gold.get("gold_chunk_ids", []))
            cands = candidates_data.get(eval_id, {}).get("candidate_chunks", [])
            chunk_texts = [c["chunk_text"] for c in cands if c.get("chunk_id") in gold_ids]
            context = "\n\n---\n\n".join(chunk_texts) if chunk_texts else ""
            user_prompt = f"""Question: {question}

Gold course chunks (use only these to define keypoints):
{context[:12000]}

Extract 3–7 required keypoints. JSON only."""
        else:
            user_prompt = f"""Question: {question}

Using general domain knowledge (information science / related fields), extract 3–7 required keypoints that a correct, current answer should cover. JSON only."""

        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_KEYPOINTS},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            out = json.loads(content)
            keypoints = out.get("required_keypoints", [])
            results[eval_id] = {"required_keypoints": keypoints}
            logger.info(f"{eval_id}: {len(keypoints)} keypoints")
        except Exception as e:
            logger.error(f"{eval_id}: {e}")
            results[eval_id] = {"required_keypoints": []}

    with open(KEYPOINTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {KEYPOINTS_PATH}")


if __name__ == "__main__":
    main()

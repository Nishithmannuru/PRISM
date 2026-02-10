"""
Step 3 — Select gold chunks (course_based only).

For each course_based question: provide question + top-15 candidate chunks to an LLM,
ask for the minimum set of chunk IDs (1–3) that fully support answering the question.
Saves gold_chunks (list of chunk_ids) as evidence ground truth.

Requires: candidates_INFO4100.json (run retrieve_candidates.py first).

Usage (from project root):
  python -m evaluation.select_gold_chunks
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

CANDIDATES_PATH = Path(__file__).parent / "candidates_INFO4100.json"
GOLD_CHUNKS_PATH = Path(__file__).parent / "gold_chunks_INFO4100.json"

SYSTEM_PROMPT = """You are preparing ground truth for an evaluation dataset.
Given a student question and a list of candidate text chunks from course materials, select the **minimum set of chunks** (by chunk_id) that together fully support answering the question.

Rules:
- Select 1 to 3 chunk IDs only.
- Chunks must directly contain the information needed to answer the question.
- Prefer fewer chunks if they suffice; do not include tangential chunks.
- Respond with valid JSON only: {"gold_chunk_ids": ["id1", "id2", ...], "reason": "brief explanation"}"""


def main():
    with open(CANDIDATES_PATH, "r") as f:
        candidates_data = json.load(f)

    client = OpenAI(api_key=OPENAI_API_KEY)
    results = {}

    for eval_id, data in candidates_data.items():
        question = data["question"]
        chunks = data.get("candidate_chunks", [])
        if not chunks:
            logger.warning(f"{eval_id}: no candidates, skipping")
            results[eval_id] = {"gold_chunk_ids": [], "reason": "no candidates"}
            continue

        chunks_text = "\n\n".join(
            f"[chunk_id: {c['chunk_id']}]\n{c['chunk_text'][:800]}"
            for c in chunks
        )
        user_prompt = f"""Student question: {question}

Candidate chunks (up to 15):
{chunks_text}

Select the minimum set of chunk_ids (1–3) that fully support answering this question. JSON only."""

        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content
            # Strip markdown code blocks if present
            if "```" in content:
                content = content.split("```")[1].replace("json", "").strip()
            out = json.loads(content)
            gold_ids = out.get("gold_chunk_ids", [])
            reason = out.get("reason", "")
            results[eval_id] = {"gold_chunk_ids": gold_ids, "reason": reason}
            logger.info(f"{eval_id}: selected {len(gold_ids)} gold chunks")
        except Exception as e:
            logger.error(f"{eval_id}: {e}")
            results[eval_id] = {"gold_chunk_ids": [], "reason": str(e)}

    with open(GOLD_CHUNKS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {GOLD_CHUNKS_PATH}")


if __name__ == "__main__":
    main()

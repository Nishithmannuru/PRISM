# PRISM Evaluation — Golden Copy Dataset Preparation

This folder holds **dataset construction only** for the PRISM evaluation: question set, candidate retrieval, gold chunk selection, keypoints, expected behavior, and the final JSONL. No scoring or model runs here.

---

## Steps (sequential)

| Step | Description | Script / Input |
|------|-------------|----------------|
| **1** | Fix question set: ~40 questions with category (`vague`, `out_of_scope`, `web_required`, `course_based`) | `questions_INFO4100.json` |
| **2** | Retrieve top-15 candidate chunks for **course_based** only | `retrieve_candidates.py` → `candidates_INFO4100.json` |
| **3** | Select 1–3 gold chunk IDs per **course_based** question (LLM) | `select_gold_chunks.py` → `gold_chunks_INFO4100.json` |
| **4** | Extract 3–7 required keypoints for **course_based** (from gold chunks) and **web_required** (general) | `extract_keypoints.py` → `keypoints_INFO4100.json` |
| **5** | Attach expected behavior per category (route, clarify, refuse, tools) | `build_dataset.py` |
| **6** | Attach personalization targets (undergrad / masters / phd grade bands) for answerable questions | `build_dataset.py` |
| **7** | Write one JSON object per question as JSONL | `build_dataset.py` → `course_eval_dataset.jsonl` |

---

## Expected behavior (Step 5)

| Category     | expected_route | needs_clarification | should_refuse | expected_tools       |
| ----------- | -------------- | ------------------- | ------------- | -------------------- |
| vague       | clarify        | true                | false         | []                   |
| out_of_scope| refuse         | false               | true          | []                   |
| web_required| web            | false               | false         | ["web_search"]       |
| course_based| course         | false               | false         | ["vector_retrieval"] |

---

## Usage (from project root)

```bash
# Step 2 — retrieve candidate chunks (course_based only)
python -m evaluation.retrieve_candidates

# Step 3 — select gold chunks (LLM; requires OPENAI_API_KEY)
python -m evaluation.select_gold_chunks

# Step 4 — extract required keypoints (LLM)
python -m evaluation.extract_keypoints

# Steps 5–7 — attach behavior + personalization, write JSONL
python -m evaluation.build_dataset
```

Output: `evaluation/course_eval_dataset.jsonl` (one JSON object per line).

---

## Files

- `questions_INFO4100.json` — Question set with `eval_id`, `category`, `question`.
- `candidates_INFO4100.json` — Per-eval_id candidate chunks (chunk_id, doc_id, page_or_slide, chunk_text).
- `gold_chunks_INFO4100.json` — Per-eval_id gold chunk IDs (1–3) for course_based.
- `keypoints_INFO4100.json` — Per-eval_id required_keypoints for course_based + web_required.
- `course_eval_dataset.jsonl` — Full 40-question eval dataset (metadata, expected_behavior, retrieval info, keypoints, personalization_targets).
- `course_eval_dataset_10q.jsonl` — **Reduced 10-question dataset** (2 vague, 1 out_of_scope, 2 web_required, 5 course_based) for faster eval runs (~1.5–2.5 hours instead of 9–10).

---

## Faster eval run (10 questions)

To run the eval pipeline on the reduced set:

```bash
python scripts/run_eval.py --dataset_path evaluation/course_eval_dataset_10q.jsonl --course_id INFO4100 --results_dir results/eval --workers 4
```

Total runs: 10 × 5 variants × 3 profiles = 150 (vs 600 for the full set).

---

## Cursor prompt (dataset preparation only)

Use this when adding a **new course** or **new question list**: provide course name + question list with categories, then ask Cursor to generate one JSON object per question following the schema above (eval_id, course_id, category, question, expected_behavior, optional candidate_chunks/gold_chunks placeholders, required_keypoints for answerable, clarification_requirements for vague, refusal_requirements for out_of_scope, personalization_targets for answerable). Output valid JSON only; no explanations.

---

## Confirmation

At this stage you are **not** scoring, **not** running PRISM, and **only** preparing ground-truth constraints. Once this dataset exists, downstream evaluation (DeepEval, RAGAS, agent metrics, ablations) is straightforward and defensible.

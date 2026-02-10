# PRISM Evaluation Runner

Reproducible evaluation pipeline: load JSONL eval dataset, run ablation variants, score with DeepEval/RAGAS/LLM-judge (Claude), and write tables/plots.

## Usage (from project root)

```bash
python scripts/run_eval.py \
  --dataset_path evaluation/course_eval_dataset.jsonl \
  --course_id INFO4100 \
  --results_dir results/eval
```

Optional: `--variants`, `--profiles`, `--max_items`, `--workers` (default 4; parallel metrics only; variant runs are sequential to avoid LangGraph threading issues).

## Variants

- **llm_only**: No retrieval, no tools, no personalization; OpenAI with question only.
- **retriever_only**: Vector retrieval returns top-15 chunks; answer is trivial list of chunks.
- **no_personalization**: Full graph with degree/major adaptation disabled.
- **no_internal_eval**: Full graph with evaluation/refinement loop skipped.
- **full_system**: PRISM as-is.

## Environment

- **OPENAI_API_KEY**: PRISM and llm_only variant.
- **PINECONE_API_KEY** (and index): retriever_only and course_based runs.
- **ANTHROPIC_API_KEY**: Claude judge (correctness, groundedness, clarification/refusal, source_credibility, bias, toxicity).
- **TAVILY_API_KEY**: Web search for web_required questions.

## Outputs under `results_dir`

- **raw/runs.jsonl**: Per-run traces (eval_id, variant, profile, question, record, trace).
- **raw/scored_runs.jsonl**: Same plus `scores` dict per run.
- **tables/**: CSV (per_variant_overall, per_category, deltas_full_system_minus_variant).
- **plots/**: PNG and PDF bar charts (overall per metric, per-category top metrics).
- **cache/judge_cache.jsonl**: Claude judge cache keyed by (metric, eval_id, variant, profile, answer_hash, context_hash).
- **summary.json**: Timestamp, git hash, n_runs, overall_means.

## Metrics (by category)

- **vague**: clarification_quality, tool_correctness, task_completeness.
- **out_of_scope**: refusal_correctness, tool_correctness, task_completeness.
- **course_based / web_required**: correctness (keypoints), readability (band alignment), bias, toxicity, context_recall/precision/relevancy, groundedness, tool_correctness, task_completeness; **web_required** also source_credibility.

Correctness is LLM-judge against `required_keypoints`; no single gold paragraph. Tool correctness compares `tools_used` to `expected_tools` from dataset.

"""GPT-4.1-mini only scoring for all PRISM runs.

Skips Anthropic judge — runs only GPT-4.1-mini to complete all metric evaluations.
Leverages existing GPT cache to skip already-scored metrics.

Optimized: parallelizes metric groups within each run for 3-4x speedup.
"""

import sys
import json
import time
import hashlib
import threading
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from prism_eval.config import (
    DATASET_PATH, RAW_DIR, CACHE_DIR,
    METRICS_BY_CATEGORY, MAX_WORKERS,
)
from prism_eval.metrics.deepeval_metrics import GPT41MiniJudge, PRISMMetrics, ScoreCache

_write_lock = threading.Lock()
_cache_lock = threading.Lock()

# Shared cache and judge — thread-safe because ScoreCache uses file append
# and GPT41MiniJudge creates new API calls per request
_shared_cache = None
_shared_judge = None


def get_shared():
    """Get shared cache and judge (initialized once)."""
    global _shared_cache, _shared_judge
    if _shared_cache is None:
        _shared_cache = ScoreCache(CACHE_DIR / "judge_cache_gpt41mini.jsonl")
        _shared_judge = GPT41MiniJudge()
    return _shared_cache, _shared_judge


def make_metrics():
    """Create a new PRISMMetrics instance (per-task to avoid state issues)."""
    cache, judge = get_shared()
    return PRISMMetrics(judge, "gpt-4.1-mini", cache)


def load_dataset(path):
    records = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            records[rec["eval_id"]] = rec
    return records


def load_unique_runs(path):
    runs = []
    seen = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = (r.get("eval_id"), r.get("variant"), r.get("profile"))
            if key not in seen:
                seen.add(key)
                runs.append(r)
    return runs


def load_existing_scored(path):
    """Load existing scored runs, return set of keys."""
    scored = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    s = json.loads(line)
                    key = (s["eval_id"], s["variant"], s["profile"])
                    scored.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
    return scored


def score_single(run, record):
    """Score a single run with GPT-4.1-mini, parallelizing metric groups."""
    metrics = make_metrics()
    category = record.get("category", "")
    test_case = metrics.build_test_case(run, record)

    all_scores = {}

    if category in ("course_based", "web_required", "multi_hop"):
        # Parallelize 4 metric groups: RAG, Agent, Response, Safety
        with ThreadPoolExecutor(max_workers=4) as inner:
            futures = {
                inner.submit(metrics.compute_rag_metrics, test_case, run, record): "rag",
                inner.submit(metrics.compute_agent_metrics, test_case, run, record): "agent",
                inner.submit(metrics.compute_response_metrics, test_case, run, record): "response",
                inner.submit(metrics.compute_safety_metrics, test_case, run, record): "safety",
            }
            for f in as_completed(futures):
                group = futures[f]
                try:
                    all_scores.update(f.result(timeout=120))
                except Exception as e:
                    print(f"    Warning: {group} metrics failed for {run.get('eval_id')}: {e}")
    elif category == "vague":
        # Agent + Safety (parallelize)
        with ThreadPoolExecutor(max_workers=2) as inner:
            futures = {
                inner.submit(metrics.compute_agent_metrics, test_case, run, record): "agent",
                inner.submit(metrics.compute_safety_metrics, test_case, run, record): "safety",
            }
            for f in as_completed(futures):
                try:
                    all_scores.update(f.result(timeout=120))
                except Exception as e:
                    print(f"    Warning: metrics failed for {run.get('eval_id')}: {e}")
    elif category == "out_of_scope":
        with ThreadPoolExecutor(max_workers=2) as inner:
            futures = {
                inner.submit(metrics.compute_agent_metrics, test_case, run, record): "agent",
                inner.submit(metrics.compute_safety_metrics, test_case, run, record): "safety",
            }
            for f in as_completed(futures):
                try:
                    all_scores.update(f.result(timeout=120))
                except Exception as e:
                    print(f"    Warning: metrics failed for {run.get('eval_id')}: {e}")

    return {
        "eval_id": run["eval_id"],
        "variant": run["variant"],
        "profile": run["profile"],
        "course_id": run.get("course_id", ""),
        "category": record.get("category", ""),
        "question": run.get("question", record.get("question", "")),
        "scores": {
            "gpt41mini": all_scores,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="GPT-4.1-mini only scoring")
    parser.add_argument("--workers", type=int, default=24, help="Outer workers (runs in parallel)")
    parser.add_argument("--dry-run", action="store_true", help="Just count what needs scoring")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...", flush=True)
    dataset = load_dataset(DATASET_PATH)
    runs = load_unique_runs(RAW_DIR / "runs.jsonl")
    print(f"  {len(runs)} unique runs, {len(dataset)} dataset records", flush=True)

    # Initialize shared cache
    get_shared()
    print(f"  {len(_shared_cache._cache)} cached GPT scores", flush=True)

    # Load existing scored runs
    scored_path = RAW_DIR / "scored_runs_gpt.jsonl"
    existing_scored = load_existing_scored(scored_path)
    print(f"  {len(existing_scored)} existing scored runs in {scored_path.name}", flush=True)

    # Build task list
    tasks = []
    skipped_no_answer = 0
    skipped_scored = 0

    for run in runs:
        eid = run.get("eval_id", "")
        variant = run.get("variant", "")
        profile = run.get("profile", "")
        record = dataset.get(eid)

        if not record:
            continue

        key = (eid, variant, profile)
        if key in existing_scored:
            skipped_scored += 1
            continue

        answer = run.get("final_answer", "")
        if not answer:
            skipped_no_answer += 1
            continue

        tasks.append((run, record))

    print(f"\n  Tasks to score: {len(tasks)}", flush=True)
    print(f"  Skipped (already scored): {skipped_scored}", flush=True)
    print(f"  Skipped (no answer): {skipped_no_answer}", flush=True)

    if args.dry_run:
        print("\nDry run — exiting.", flush=True)
        return

    if not tasks:
        print("\nAll scoring complete!", flush=True)
        return

    print(f"\nScoring {len(tasks)} runs with {args.workers} outer workers + 4 inner workers per run (GPT-4.1-mini)...", flush=True)

    f_out = open(scored_path, "a")
    completed = 0
    errors = 0
    total = len(tasks)
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(score_single, run, rec): (run, rec) for run, rec in tasks}
        for future in as_completed(futures):
            try:
                scored_run = future.result(timeout=300)
                with _write_lock:
                    f_out.write(json.dumps(scored_run, default=str) + "\n")
                    f_out.flush()
            except Exception as e:
                run, rec = futures[future]
                errors += 1
                print(f"  ERROR {run.get('eval_id')}/{run.get('variant')}/{run.get('profile')}: {e}", flush=True)

            completed += 1
            if completed % 20 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate / 60 if rate > 0 else 0
                print(f"  Progress: {completed}/{total} ({completed*100//total}%) "
                      f"| {rate:.2f}/sec | ETA: {eta:.0f}min | Errors: {errors}", flush=True)

    f_out.close()
    elapsed = time.time() - start_time
    print(f"\nDone! Scored {completed - errors}/{total} runs in {elapsed/60:.1f} min. Errors: {errors}", flush=True)
    print(f"Output: {scored_path}", flush=True)


if __name__ == "__main__":
    main()

"""Main evaluation runner for PRISM.

Orchestrates:
1. Loading the dataset
2. Running PRISM pipeline in parallel (separate graph per worker)
3. Running baseline via OpenAI Batch API
4. Computing metrics with dual judges in parallel
5. Aggregating scores
6. Generating reports (tables, plots, LaTeX)
"""

import sys
import json
import time
import argparse
import traceback
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from prism_eval.config import (
    DATASET_PATH, RESULTS_DIR, RAW_DIR, TABLES_DIR, PLOTS_DIR, LATEX_DIR, CACHE_DIR,
    VARIANTS, PROFILES, METRICS_BY_CATEGORY, ALL_METRICS,
    MAX_WORKERS, TIMEOUT_PER_QUERY, OPENAI_API_KEY, SYSTEM_MODEL,
    FULL_PROFILE_VARIANTS, SINGLE_PROFILE_VARIANTS,
)
from prism_eval.runners.prism_wrapper import PRISMEvalWrapper, RunTrace
from prism_eval.metrics.dual_judge import DualJudgeEvaluator
from prism_eval.reporting.aggregate import aggregate_scores
from prism_eval.reporting.tables import write_tables
from prism_eval.reporting.plots import write_plots


# Thread-local storage for per-worker PRISM wrappers
_thread_local = threading.local()
_write_lock = threading.Lock()


def _get_wrapper():
    """Get or create a per-thread PRISMEvalWrapper (each has its own graph instances)."""
    if not hasattr(_thread_local, "wrapper"):
        _thread_local.wrapper = PRISMEvalWrapper()
    return _thread_local.wrapper


# ── Dataset Loading ────────────────────────────────────────────────────────

def load_dataset(path: Path, max_items: int = 0, course_id: str = None) -> list:
    records = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if course_id and r.get("course_id") != course_id:
                continue
            records.append(r)
            if max_items and len(records) >= max_items:
                break
    return records


# ── Phase 1a: Baseline via OpenAI Batch API ───────────────────────────────

def run_baseline_batch(records: list, profiles: dict, output_path: Path) -> list:
    """Run baseline (plain LLM) via OpenAI Batch API for 50% cost savings."""
    from openai import OpenAI
    import tempfile

    client = OpenAI(api_key=OPENAI_API_KEY)
    traces = []

    # Check existing runs
    existing_keys = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    t = json.loads(line)
                    if t.get("variant") == "baseline":
                        key = f"{t['eval_id']}|baseline|{t['profile']}"
                        existing_keys.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue

    # Build batch requests
    batch_requests = []
    request_map = {}  # custom_id -> (record, profile_name)

    for record in records:
        for profile_name in profiles:
            key = f"{record['eval_id']}|baseline|{profile_name}"
            if key in existing_keys:
                continue

            custom_id = f"{record['eval_id']}__baseline__{profile_name}"
            batch_requests.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": SYSTEM_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a helpful teaching assistant. Answer the student's "
                                "question about their course material clearly and accurately. "
                                f"The course is: {record['course_id']}."
                            ),
                        },
                        {"role": "user", "content": record["question"]},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 2000,
                },
            })
            request_map[custom_id] = (record, profile_name)

    if not batch_requests:
        print("  All baseline runs already cached.")
        return traces

    print(f"  Submitting {len(batch_requests)} baseline requests via OpenAI Batch API...")

    # Write batch input file
    batch_input_path = CACHE_DIR / "baseline_batch_input.jsonl"
    with open(batch_input_path, "w") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")

    # Upload and create batch
    with open(batch_input_path, "rb") as f:
        input_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )

    print(f"  Batch created: {batch.id}. Polling for completion...")

    # Poll for completion
    while True:
        batch_status = client.batches.retrieve(batch.id)
        status = batch_status.status
        completed = batch_status.request_counts.completed
        total = batch_status.request_counts.total
        failed = batch_status.request_counts.failed

        if status == "completed":
            print(f"  Batch complete: {completed}/{total} succeeded, {failed} failed.")
            break
        elif status in ("failed", "expired", "cancelled"):
            print(f"  Batch {status}. Falling back to sequential baseline.")
            return _run_baseline_sequential(records, profiles, output_path, existing_keys)

        print(f"  Batch status: {status} ({completed}/{total})...")
        time.sleep(10)

    # Download results
    output_file = client.files.content(batch_status.output_file_id)
    results_text = output_file.text

    f_out = open(output_path, "a")
    for line in results_text.strip().split("\n"):
        try:
            result = json.loads(line)
            custom_id = result["custom_id"]
            record, profile_name = request_map[custom_id]

            response_body = result.get("response", {}).get("body", {})
            answer = ""
            if response_body.get("choices"):
                answer = response_body["choices"][0]["message"]["content"]

            trace = RunTrace(
                eval_id=record["eval_id"],
                variant="baseline",
                profile=profile_name,
                course_id=record["course_id"],
                question=record["question"],
                final_answer=answer,
                route_taken="baseline_direct",
                tools_used=[],
                source_type="baseline",
            )
            trace_dict = trace.to_dict()
            with _write_lock:
                f_out.write(json.dumps(trace_dict) + "\n")
                f_out.flush()
            traces.append(trace_dict)
        except Exception as e:
            print(f"  Error processing batch result: {e}")

    f_out.close()
    print(f"  Baseline batch: {len(traces)} runs saved.")
    return traces


def _run_baseline_sequential(records, profiles, output_path, existing_keys):
    """Fallback: run baseline sequentially if batch fails."""
    wrapper = PRISMEvalWrapper()
    traces = []
    f_out = open(output_path, "a")

    for record in records:
        for profile_name, user_ctx in profiles.items():
            key = f"{record['eval_id']}|baseline|{profile_name}"
            if key in existing_keys:
                continue
            trace = wrapper.run(
                question=record["question"],
                course_id=record["course_id"],
                user_context=user_ctx,
                variant="baseline",
                eval_id=record["eval_id"],
                profile=profile_name,
            )
            trace_dict = trace.to_dict()
            f_out.write(json.dumps(trace_dict) + "\n")
            f_out.flush()
            traces.append(trace_dict)

    f_out.close()
    return traces


# ── Phase 1b: PRISM pipeline runs in parallel ─────────────────────────────

def run_prism_variants(
    records: list,
    variants: list,
    profiles: dict,
    output_path: Path,
    max_workers: int = 4,
) -> list:
    """Run PRISM pipeline for all (record, variant, profile) combinations in parallel.

    Each worker gets its own PRISMEvalWrapper with independent graph instances,
    so LangGraph thread-safety is not an issue.
    """
    all_traces = []

    # Check existing runs
    existing_keys = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    t = json.loads(line)
                    key = f"{t['eval_id']}|{t['variant']}|{t['profile']}"
                    existing_keys.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"  Found {len(existing_keys)} existing runs.")

    # Build task list (exclude baseline — handled by batch API)
    # Optimize: only full_system/no_personalization/baseline need all profiles
    prism_variants = [v for v in variants if v != "baseline"]
    tasks = []
    for record in records:
        for variant in prism_variants:
            variant_profiles = profiles if variant in FULL_PROFILE_VARIANTS else {"undergrad": profiles["undergrad"]}
            for profile_name, user_ctx in variant_profiles.items():
                key = f"{record['eval_id']}|{variant}|{profile_name}"
                if key in existing_keys:
                    continue
                tasks.append((record, variant, profile_name, user_ctx))

    if not tasks:
        print("  All PRISM runs already cached.")
        return all_traces

    total = len(tasks)
    print(f"  Running {total} PRISM pipeline executions with {max_workers} workers...")

    completed = 0
    f_out = open(output_path, "a")
    start_time = time.time()

    def execute_single(args):
        record, variant, profile_name, user_ctx = args
        wrapper = _get_wrapper()
        return wrapper.run(
            question=record["question"],
            course_id=record["course_id"],
            user_context=user_ctx,
            variant=variant,
            eval_id=record["eval_id"],
            profile=profile_name,
        ).to_dict()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(execute_single, args): args for args in tasks}
        for future in as_completed(futures):
            try:
                trace_dict = future.result(timeout=TIMEOUT_PER_QUERY * 3)
            except Exception as e:
                args = futures[future]
                record, variant, profile_name, _ = args
                trace_dict = RunTrace(
                    eval_id=record["eval_id"],
                    variant=variant,
                    profile=profile_name,
                    course_id=record["course_id"],
                    question=record["question"],
                    error=str(e),
                ).to_dict()

            with _write_lock:
                f_out.write(json.dumps(trace_dict) + "\n")
                f_out.flush()
            all_traces.append(trace_dict)

            completed += 1
            if completed % 20 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate / 60 if rate > 0 else 0
                print(f"  Progress: {completed}/{total} ({completed*100//total}%) "
                      f"| {rate:.1f} runs/sec | ETA: {eta:.0f}min")

    f_out.close()
    print(f"  Phase 1 complete: {len(all_traces)} new runs ({len(existing_keys)} from cache).")
    return all_traces


# ── Phase 2: Compute metrics in parallel ──────────────────────────────────

def compute_all_metrics(
    traces: list,
    records: list,
    output_path: Path,
    max_workers: int = 4,
) -> list:
    """Compute all metrics for each trace using dual judges in parallel."""
    record_map = {r["eval_id"]: r for r in records}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Check existing scored runs
    existing_keys = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    s = json.loads(line)
                    key = f"{s['eval_id']}|{s['variant']}|{s['profile']}"
                    existing_keys.add(key)
                except (json.JSONDecodeError, KeyError):
                    continue
        print(f"  Found {len(existing_keys)} existing scored runs.")

    tasks = []
    for trace in traces:
        key = f"{trace['eval_id']}|{trace['variant']}|{trace['profile']}"
        if key in existing_keys:
            continue
        record = record_map.get(trace["eval_id"])
        if record:
            tasks.append((trace, record))

    if not tasks:
        print("  All metrics already computed.")
        scored = []
        if output_path.exists():
            with open(output_path) as f:
                for line in f:
                    try:
                        scored.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return scored

    print(f"  Computing metrics for {len(tasks)} runs with {max_workers} workers + dual judges...")

    # Each worker gets its own DualJudgeEvaluator
    evaluator_local = threading.local()

    def get_evaluator():
        if not hasattr(evaluator_local, "evaluator"):
            evaluator_local.evaluator = DualJudgeEvaluator()
        return evaluator_local.evaluator

    f_out = open(output_path, "a")
    completed = 0
    total = len(tasks)
    start_time = time.time()

    def score_single(args):
        trace, record = args
        evaluator = get_evaluator()
        scores = evaluator.compute_all(trace, record)
        return {
            "eval_id": trace["eval_id"],
            "variant": trace["variant"],
            "profile": trace["profile"],
            "course_id": trace["course_id"],
            "category": record.get("category", ""),
            "question": trace.get("question", record.get("question", "")),
            "scores": scores,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(score_single, args): args for args in tasks}
        for future in as_completed(futures):
            try:
                scored_run = future.result(timeout=300)
                with _write_lock:
                    f_out.write(json.dumps(scored_run, default=str) + "\n")
                    f_out.flush()
            except Exception as e:
                args = futures[future]
                trace, record = args
                print(f"  Error scoring {trace['eval_id']}/{trace['variant']}: {e}")

            completed += 1
            if completed % 10 == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total - completed) / rate / 60 if rate > 0 else 0
                print(f"  Metrics: {completed}/{total} ({completed*100//total}%) "
                      f"| {rate:.1f}/sec | ETA: {eta:.0f}min")

    f_out.close()

    # Reload all scored runs
    scored = []
    with open(output_path) as f:
        for line in f:
            try:
                scored.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"  Phase 2 complete: {len(scored)} total scored runs.")
    return scored


# ── Phase 3: Aggregate and Report ─────────────────────────────────────────

def generate_reports(scored_runs: list, all_judge_scores: list = None):
    """Aggregate scores and generate tables/plots."""
    print("Generating reports...")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

    aggregated = aggregate_scores(scored_runs)

    write_tables(aggregated, TABLES_DIR, LATEX_DIR)
    write_plots(aggregated, PLOTS_DIR)

    # Inter-judge agreement
    if all_judge_scores:
        agreement = DualJudgeEvaluator.compute_inter_judge_agreement(all_judge_scores)
        agreement_path = TABLES_DIR / "inter_judge_agreement.json"
        with open(agreement_path, "w") as f:
            json.dump(agreement, f, indent=2)
        print(f"  Inter-judge agreement: {agreement_path}")

    # Summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_runs": len(scored_runs),
        "variants": sorted(set(r["variant"] for r in scored_runs)),
        "profiles": sorted(set(r["profile"] for r in scored_runs)),
        "courses": sorted(set(r["course_id"] for r in scored_runs)),
        "categories": sorted(set(r["category"] for r in scored_runs)),
        "overall_means": aggregated.get("overall", {}),
    }
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"  Results: {RESULTS_DIR}")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PRISM Evaluation Runner")
    parser.add_argument("--dataset", type=str, default=str(DATASET_PATH))
    parser.add_argument("--course-id", type=str, default=None)
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--profiles", nargs="+", default=list(PROFILES.keys()))
    parser.add_argument("--max-items", type=int, default=0, help="Limit dataset size (0=all)")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    parser.add_argument("--skip-runs", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--runs-only", action="store_true")
    parser.add_argument("--no-batch", action="store_true", help="Disable Batch API for baseline")
    args = parser.parse_args()

    profiles = {k: v for k, v in PROFILES.items() if k in args.profiles}

    for d in [RAW_DIR, TABLES_DIR, PLOTS_DIR, LATEX_DIR, CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.dataset}...")
    records = load_dataset(Path(args.dataset), max_items=args.max_items, course_id=args.course_id)
    print(f"Loaded {len(records)} records.")

    runs_path = RAW_DIR / "runs.jsonl"
    scored_path = RAW_DIR / "scored_runs.jsonl"

    # ── Phase 1: Pipeline runs ──
    if not args.skip_runs:
        print("\n=== Phase 1: Pipeline Runs ===")

        # 1a: Baseline via Batch API (or sequential fallback)
        if "baseline" in args.variants:
            print("\n--- Baseline runs ---")
            if args.no_batch:
                _run_baseline_sequential(records, profiles, runs_path, set())
            else:
                run_baseline_batch(records, profiles, runs_path)

        # 1b: PRISM variants in parallel
        prism_variants = [v for v in args.variants if v != "baseline"]
        if prism_variants:
            print("\n--- PRISM variant runs ---")
            run_prism_variants(
                records=records,
                variants=prism_variants,
                profiles=profiles,
                output_path=runs_path,
                max_workers=args.workers,
            )
    else:
        print("Skipping pipeline runs (--skip-runs).")

    # Load all traces
    all_traces = []
    if runs_path.exists():
        with open(runs_path) as f:
            for line in f:
                try:
                    all_traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"Total traces: {len(all_traces)}")

    if args.runs_only:
        print("Done (--runs-only).")
        return

    # ── Phase 2: Metrics ──
    if not args.skip_metrics:
        print("\n=== Phase 2: Metric Computation ===")
        scored_runs = compute_all_metrics(
            traces=all_traces,
            records=records,
            output_path=scored_path,
            max_workers=args.workers,
        )
    else:
        print("Skipping metrics (--skip-metrics).")
        scored_runs = []
        if scored_path.exists():
            with open(scored_path) as f:
                for line in f:
                    try:
                        scored_runs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    # ── Phase 3: Reports ──
    print("\n=== Phase 3: Report Generation ===")
    all_judge_scores = [
        r.get("scores", {}) for r in scored_runs
        if "sonnet" in r.get("scores", {}) and "gpt41mini" in r.get("scores", {})
    ]
    generate_reports(scored_runs, all_judge_scores)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

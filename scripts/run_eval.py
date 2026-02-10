"""
PRISM evaluation runner: run ablations, compute metrics, write tables and plots.

Usage (from project root):
  python scripts/run_eval.py --dataset_path evaluation/course_eval_dataset.jsonl --course_id INFO4100 --results_dir results/eval

Optional:
  --variants llm_only,retriever_only,no_personalization,no_internal_eval,full_system
  --profiles undergrad,masters,phd
  --max_items N
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

# Project root and load .env first so ANTHROPIC_API_KEY etc. are available for judge metrics
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    pass

from eval_runner.config import (
    VARIANTS,
    DEFAULT_PROFILES,
    ensure_results_dirs,
    RAW_DIR,
    CACHE_DIR,
    JUDGE_CACHE_FILENAME,
    SUMMARY_JSON,
)
from eval_runner.run_variants import run_all
from eval_runner.metrics.compute import compute_all_metrics
from eval_runner.reporting.aggregate import aggregate_scores
from eval_runner.reporting.tables import write_tables
from eval_runner.reporting.plots import write_plots
from eval_runner.metrics.judge_metrics import set_judge_cache_path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _git_hash() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout:
            return out.stdout.strip()[:12]
    except Exception:
        pass
    return ""


def main():
    parser = argparse.ArgumentParser(description="PRISM evaluation runner")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to eval JSONL")
    parser.add_argument("--course_id", type=str, required=True, help="Course ID (e.g. INFO4100)")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory (raw/, tables/, plots/, summary.json)")
    parser.add_argument("--variants", type=str, default=None, help="Comma-separated variants (default: all)")
    parser.add_argument("--profiles", type=str, default=None, help="Comma-separated profiles (default: undergrad,masters,phd)")
    parser.add_argument("--max_items", type=int, default=None, help="Max dataset items to run (default: all)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for runs and metrics (default: 4)")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = PROJECT_ROOT / dataset_path
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = PROJECT_ROOT / results_dir
    ensure_results_dirs(results_dir)

    variants = args.variants.split(",") if args.variants else VARIANTS
    variants = [v.strip() for v in variants if v.strip()]
    profiles = args.profiles.split(",") if args.profiles else DEFAULT_PROFILES
    profiles = [p.strip() for p in profiles if p.strip()]
    workers = max(1, int(args.workers))

    # 1) Run variants and write raw/runs.jsonl (sequential: LangGraph is not thread-safe for concurrent invoke)
    logger.info("Running variants (sequential to avoid LangGraph executor conflicts)...")
    run_all(
        dataset_path=dataset_path,
        course_id=args.course_id,
        results_dir=results_dir,
        variants=variants,
        profiles=profiles,
        max_items=args.max_items,
        workers=1,
    )

    # 2) Load runs, compute metrics, build scored_runs
    raw_path = results_dir / RAW_DIR / "runs.jsonl"
    if not raw_path.exists():
        logger.error(f"No runs file: {raw_path}")
        sys.exit(1)

    judge_cache_path = results_dir / CACHE_DIR / JUDGE_CACHE_FILENAME
    judge_cache_path.parent.mkdir(parents=True, exist_ok=True)
    set_judge_cache_path(judge_cache_path)

    # Load all run rows
    rows = []
    with open(raw_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    def compute_one(row):
        eval_id = row.get("eval_id", "")
        category = row.get("category", "")
        variant = row.get("variant", "")
        profile = row.get("profile", "")
        question = row.get("question", "")
        record = row.get("record", {})
        trace = row.get("trace", {})
        try:
            scores = compute_all_metrics(
                eval_id=eval_id,
                category=category,
                variant=variant,
                profile=profile,
                question=question,
                record=record,
                trace=trace,
                judge_cache_path=judge_cache_path,
            )
        except Exception as e:
            logger.warning(f"Metrics failed for {eval_id} {variant} {profile}: {e}")
            scores = {}
        return {
            "eval_id": eval_id,
            "course_id": row.get("course_id", ""),
            "category": category,
            "variant": variant,
            "profile": profile,
            "question": question,
            "scores": scores,
        }

    scored_runs = []
    scored_path = results_dir / RAW_DIR / "scored_runs.jsonl"
    with open(scored_path, "w") as sf_empty:
        pass  # truncate

    logger.info("Computing metrics (workers=%s)...", workers)
    if workers <= 1:
        for row in rows:
            out_row = compute_one(row)
            scored_runs.append(out_row)
            with open(scored_path, "a") as sf:
                sf.write(json.dumps(out_row, default=str) + "\n")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(compute_one, row): row for row in rows}
            for future in as_completed(futures):
                try:
                    out_row = future.result()
                    scored_runs.append(out_row)
                except Exception as e:
                    logger.warning(f"Metric task failed: {e}")
        # Write in deterministic order (eval_id, variant, profile) for reproducibility
        scored_runs.sort(key=lambda r: (r["eval_id"], r["variant"], r["profile"]))
        with open(scored_path, "w") as sf:
            for out_row in scored_runs:
                sf.write(json.dumps(out_row, default=str) + "\n")

    logger.info(f"Computed metrics for {len(scored_runs)} runs. Wrote {scored_path}")

    # 3) Aggregate
    aggregates = aggregate_scores(scored_runs)

    # 4) Tables and plots
    write_tables(aggregates, results_dir)
    write_plots(aggregates, results_dir)

    # 5) summary.json
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_hash": _git_hash(),
        "dataset_path": str(dataset_path),
        "course_id": args.course_id,
        "variants": variants,
        "profiles": profiles,
        "max_items": args.max_items,
        "n_runs": len(scored_runs),
        "overall_means": {m: aggregates.get("overall", {}).get(m, {}).get("mean") for m in aggregates.get("overall", {})},
    }
    summary_path = results_dir / SUMMARY_JSON
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Wrote {summary_path}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()

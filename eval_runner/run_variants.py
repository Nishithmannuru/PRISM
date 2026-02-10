"""
Run PRISM variants for each dataset item and save raw traces to results_dir/raw/.
Supports parallel execution via workers > 1.
"""

import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from eval_runner.config import VARIANTS, ensure_results_dirs, RAW_DIR
from eval_runner.load_dataset import load_dataset, filter_by_course, get_profiles_to_test
from eval_runner.prism_wrapper import run_prism

logger = logging.getLogger(__name__)


def _run_one(
    course_id: str,
    record: Dict[str, Any],
    variant: str,
    profile: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run a single (record, variant, profile). Returns (row for JSONL, summary)."""
    eval_id = record.get("eval_id", "")
    question = record.get("question", "")
    category = record.get("category", "")
    try:
        trace = run_prism(
            course_id=course_id,
            student_profile=profile,
            question=question,
            variant_config=variant,
        )
        row = {
            "eval_id": eval_id,
            "course_id": course_id,
            "category": category,
            "variant": variant,
            "profile": profile,
            "question": question,
            "record": record,
            "trace": trace,
        }
        summary = {"eval_id": eval_id, "variant": variant, "profile": profile, "error": trace.get("error")}
        return (row, summary)
    except Exception as e:
        logger.exception(f"Run {eval_id} {variant} {profile} failed: {e}")
        row = {
            "eval_id": eval_id,
            "course_id": course_id,
            "category": category,
            "variant": variant,
            "profile": profile,
            "question": question,
            "record": record,
            "trace": {"final_answer_text": "", "route_taken": "course", "tools_used": [], "retrieval_context": [], "web_context": [], "citations": [], "error": str(e)},
        }
        return (row, {"eval_id": eval_id, "variant": variant, "profile": profile, "error": str(e)})


def run_all(
    dataset_path: Path,
    course_id: str,
    results_dir: Path,
    variants: Optional[List[str]] = None,
    profiles: Optional[List[str]] = None,
    max_items: Optional[int] = None,
    workers: int = 1,
) -> List[Dict[str, Any]]:
    """
    For each (record, variant, profile) run PRISM and append one JSONL row to results_dir/raw/runs.jsonl.
    If workers > 1, runs in parallel with ThreadPoolExecutor.
    Returns list of run summary dicts (eval_id, variant, profile, error if any).
    """
    ensure_results_dirs(results_dir)
    raw_dir = results_dir / RAW_DIR
    out_path = raw_dir / "runs.jsonl"
    with open(out_path, "w") as _:
        pass  # truncate for fresh run

    records = load_dataset(dataset_path, max_items=max_items)
    records = filter_by_course(records, course_id)
    if not records:
        logger.warning(f"No records for course_id={course_id}")
        return []

    variants = variants or VARIANTS
    runs_done: List[Dict[str, Any]] = []

    # Build flat list of (record, variant, profile)
    tasks: List[Tuple[Dict[str, Any], str, str]] = []
    for record in records:
        eval_id = record.get("eval_id", "")
        category = record.get("category", "")
        profs = profiles or get_profiles_to_test(record)
        for variant in variants:
            for profile in profs:
                tasks.append((record, variant, profile))

    if workers <= 1:
        for record, variant, profile in tasks:
            row, summary = _run_one(course_id, record, variant, profile)
            with open(out_path, "a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            runs_done.append(summary)
            logger.info(f"Run {summary['eval_id']} {variant} {profile} -> ok")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        write_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_task = {
                executor.submit(_run_one, course_id, record, variant, profile): (record, variant, profile)
                for record, variant, profile in tasks
            }
            for future in as_completed(future_to_task):
                record, variant, profile = future_to_task[future]
                try:
                    row, summary = future.result()
                    with write_lock:
                        with open(out_path, "a") as f:
                            f.write(json.dumps(row, default=str) + "\n")
                        runs_done.append(summary)
                    logger.info(f"Run {summary['eval_id']} {variant} {profile} -> ok")
                except Exception as e:
                    logger.exception(f"Task failed: {e}")

    logger.info(f"Wrote {len(runs_done)} runs to {out_path}")
    return runs_done

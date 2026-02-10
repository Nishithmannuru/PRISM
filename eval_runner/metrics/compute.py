"""
Compute all applicable metrics for a single run (row from run_variants).
Uses router to decide which metrics apply; fills scores dict.
"""

import logging
from typing import Dict, Any, List

from eval_runner.metrics.router import (
    metrics_for_category_and_trace,
    CORRECTNESS,
    READABILITY,
    BIAS,
    TOXICITY,
    CONTEXT_RECALL,
    CONTEXT_PRECISION,
    CONTEXT_RELEVANCY,
    SOURCE_CREDIBILITY,
    GROUNDEDNESS,
    TOOL_CORRECTNESS,
    TASK_COMPLETENESS,
    CLARIFICATION_QUALITY,
    REFUSAL_CORRECTNESS,
)
from eval_runner.config import CATEGORY_EXPECTED_TOOLS

logger = logging.getLogger(__name__)


def tool_correctness(trace: Dict[str, Any], expected_tools: List[str]) -> float:
    """1.0 if tools_used matches expected_tools, else 0.0 (or partial)."""
    used = set(trace.get("tools_used") or [])
    expected = set(expected_tools or [])
    if not expected and not used:
        return 1.0
    if not expected:
        return 1.0 if not used else 0.0
    return 1.0 if used == expected else (len(used & expected) / len(expected) if expected else 1.0)


def task_completeness(category: str, trace: Dict[str, Any], record: Dict[str, Any]) -> float:
    """Heuristic: did the system do the right thing (clarify/refuse/answer with citations)? 0-1."""
    route = trace.get("route_taken") or "course"
    expected_route = (record.get("expected_behavior") or {}).get("expected_route") or "course"
    if route != expected_route:
        return 0.0
    answer = (trace.get("final_answer_text") or "").strip()
    if category == "vague":
        return 1.0 if "?" in answer or "clarif" in answer.lower() or "more" in answer.lower() else 0.5
    if category == "out_of_scope":
        return 1.0 if answer and len(answer) > 20 else 0.5
    if category in ("course_based", "web_required"):
        return 1.0 if answer and len(answer) > 50 else 0.5
    return 0.5


def compute_all_metrics(
    eval_id: str,
    category: str,
    variant: str,
    profile: str,
    question: str,
    record: Dict[str, Any],
    trace: Dict[str, Any],
    judge_cache_path: Any = None,
) -> Dict[str, float]:
    """
    Compute all metrics that apply for this run. Returns dict metric_name -> score.
    """
    if judge_cache_path:
        from eval_runner.metrics.judge_metrics import set_judge_cache_path
        set_judge_cache_path(judge_cache_path)

    metrics_to_run = metrics_for_category_and_trace(category, trace)
    scores: Dict[str, float] = {}
    answer = (trace.get("final_answer_text") or "").strip()
    expected_behavior = record.get("expected_behavior") or {}
    expected_tools = expected_behavior.get("expected_tools") or []
    personalization_targets = record.get("personalization_targets") or {}

    for m in metrics_to_run:
        try:
            if m == CORRECTNESS:
                keypoints = record.get("required_keypoints") or []
                from eval_runner.metrics.judge_metrics import judge_correctness
                scores[m] = judge_correctness(eval_id, variant, profile, question, answer, keypoints, record)
            elif m == READABILITY:
                from eval_runner.metrics.readability_metrics import readability_band_alignment
                scores[m] = readability_band_alignment(answer, profile, personalization_targets)
            elif m == BIAS:
                from eval_runner.metrics.judge_metrics import judge_bias
                scores[m] = judge_bias(eval_id, variant, profile, answer)
            elif m == TOXICITY:
                from eval_runner.metrics.judge_metrics import judge_toxicity
                scores[m] = judge_toxicity(eval_id, variant, profile, answer)
            elif m == CONTEXT_RECALL:
                from eval_runner.metrics.deepeval_metrics import context_recall
                ctxs = [c.get("text", "") for c in (trace.get("retrieval_context") or trace.get("web_context") or [])]
                scores[m] = context_recall(question, answer, ctxs)
            elif m == CONTEXT_PRECISION:
                from eval_runner.metrics.deepeval_metrics import context_precision
                ctxs = [c.get("text", "") for c in (trace.get("retrieval_context") or trace.get("web_context") or [])]
                scores[m] = context_precision(question, answer, ctxs)
            elif m == CONTEXT_RELEVANCY:
                from eval_runner.metrics.deepeval_metrics import context_relevancy
                ctxs = [c.get("text", "") for c in (trace.get("retrieval_context") or trace.get("web_context") or [])]
                scores[m] = context_relevancy(question, ctxs)
            elif m == SOURCE_CREDIBILITY:
                from eval_runner.metrics.judge_metrics import judge_source_credibility
                scores[m] = judge_source_credibility(eval_id, variant, profile, answer, trace.get("web_context") or [])
            elif m == GROUNDEDNESS:
                from eval_runner.metrics.judge_metrics import judge_groundedness
                ctx_text = "\n\n".join([c.get("text", "") for c in (trace.get("retrieval_context") or trace.get("web_context") or [])])
                scores[m] = judge_groundedness(eval_id, variant, profile, answer, ctx_text)
            elif m == TOOL_CORRECTNESS:
                scores[m] = tool_correctness(trace, expected_tools)
            elif m == TASK_COMPLETENESS:
                scores[m] = task_completeness(category, trace, record)
            elif m == CLARIFICATION_QUALITY:
                from eval_runner.metrics.judge_metrics import judge_clarification_quality
                scores[m] = judge_clarification_quality(eval_id, variant, profile, question, answer, record)
            elif m == REFUSAL_CORRECTNESS:
                from eval_runner.metrics.judge_metrics import judge_refusal_correctness
                scores[m] = judge_refusal_correctness(eval_id, variant, profile, question, answer, record)
        except Exception as e:
            logger.warning(f"Metric {m} failed: {e}")
            scores[m] = 0.0

    return scores

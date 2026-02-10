"""Decide which metrics apply for a given category and trace (e.g. context metrics only when retrieval_context/web_context exists)."""

from typing import List, Dict, Any

# Metric names used across runner
CORRECTNESS = "correctness"
READABILITY = "readability"
BIAS = "bias"
TOXICITY = "toxicity"
CONTEXT_RECALL = "context_recall"
CONTEXT_PRECISION = "context_precision"
CONTEXT_RELEVANCY = "context_relevancy"
SOURCE_CREDIBILITY = "source_credibility"
GROUNDEDNESS = "groundedness"
TOOL_CORRECTNESS = "tool_correctness"
TASK_COMPLETENESS = "task_completeness"
CLARIFICATION_QUALITY = "clarification_quality"
REFUSAL_CORRECTNESS = "refusal_correctness"


def metrics_for_category_and_trace(category: str, trace: Dict[str, Any]) -> List[str]:
    """
    Return list of metric names that apply.
    - context_* only when retrieval_context or web_context exists.
    - source_credibility only for web_required when web_context exists.
    - clarification_quality for vague; refusal_correctness for out_of_scope.
    """
    out = [CORRECTNESS, READABILITY, BIAS, TOXICITY, TOOL_CORRECTNESS, TASK_COMPLETENESS]
    has_retrieval = bool(trace.get("retrieval_context"))
    has_web = bool(trace.get("web_context"))

    if category in ("course_based", "web_required"):
        # Keep correctness, readability, bias, toxicity, tool, task
        if has_retrieval or has_web:
            out.extend([CONTEXT_RECALL, CONTEXT_PRECISION, CONTEXT_RELEVANCY, GROUNDEDNESS])
        if category == "web_required" and has_web:
            out.append(SOURCE_CREDIBILITY)
    elif category == "vague":
        out = [CLARIFICATION_QUALITY, TOOL_CORRECTNESS, TASK_COMPLETENESS]
    elif category == "out_of_scope":
        out = [REFUSAL_CORRECTNESS, TOOL_CORRECTNESS, TASK_COMPLETENESS]

    return out

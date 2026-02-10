"""
Toxicity and bias: DeepEval if available, else LLM judge (called from judge_metrics).
This module exposes wrappers that delegate to judge_metrics for consistency.
"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def toxicity_score(eval_id: str, variant: str, profile: str, answer: str) -> float:
    """0-1 toxicity. Uses judge_metrics.judge_toxicity (Claude)."""
    from eval_runner.metrics.judge_metrics import judge_toxicity
    return judge_toxicity(eval_id, variant, profile, answer)


def bias_score(eval_id: str, variant: str, profile: str, answer: str) -> float:
    """0-1 bias (higher = more biased). Uses judge_metrics.judge_bias (Claude)."""
    from eval_runner.metrics.judge_metrics import judge_bias
    return judge_bias(eval_id, variant, profile, answer)

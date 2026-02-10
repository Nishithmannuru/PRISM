"""
DeepEval metrics: context recall, context precision, faithfulness if available.
Stub: return 0.5 when DeepEval not configured; implement when DEEPEVAL_API_KEY or package available.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


def context_recall(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
) -> float:
    """0-1 context recall. Stub: 0.5."""
    return 0.5


def context_precision(
    question: str,
    answer: str,
    contexts: List[str],
) -> float:
    """0-1 context precision. Stub: 0.5."""
    return 0.5


def context_relevancy(question: str, contexts: List[str]) -> float:
    """0-1 context relevancy. Stub: 0.5."""
    return 0.5


def faithfulness(answer: str, contexts: List[str]) -> float:
    """0-1 faithfulness. Stub: 0.5."""
    return 0.5

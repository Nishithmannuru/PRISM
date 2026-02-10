"""
RAGAS metrics: context recall, context precision, context relevancy.
Stub: return 0.5 when RAGAS not configured; implement when package/API available.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def context_recall(ground_truth: str, contexts: List[str], answer: str) -> float:
    """0-1. Stub: 0.5."""
    return 0.5


def context_precision(question: str, contexts: List[str], answer: str) -> float:
    """0-1. Stub: 0.5."""
    return 0.5


def context_relevancy(question: str, contexts: List[str]) -> float:
    """0-1. Stub: 0.5."""
    return 0.5

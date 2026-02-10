"""
Readability: Flesch-Kincaid, SMOG, Fog, Dale-Chall; grade estimate and band alignment to personalization_targets.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def _grade_level(text: str) -> Optional[float]:
    """Composite grade level from Flesch-Kincaid (US grade). Returns None if text too short."""
    if not text or len(text.strip()) < 50:
        return None
    try:
        import textstat
        fk = textstat.flesch_kincaid_grade(text)
        return float(fk)
    except Exception as e:
        logger.warning(f"textstat failed: {e}")
        return None


def readability_grade(text: str) -> Optional[float]:
    """Return US grade level (e.g. 10.5). None if unavailable."""
    return _grade_level(text)


def readability_band_alignment(
    text: str,
    profile: str,
    personalization_targets: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Score 0-1: how well the text's grade level aligns with the profile's target band.
    personalization_targets e.g. {"undergrad": {"target_grade_band": [9, 12]}, ...}
    """
    grade = _grade_level(text)
    if grade is None:
        return 0.5
    targets = personalization_targets or {}
    band = targets.get(profile, {}).get("target_grade_band")
    if not band or len(band) < 2:
        return 0.5
    low, high = band[0], band[1]
    if low <= grade <= high:
        return 1.0
    if low - 2 <= grade <= high + 2:
        return 0.7
    if low - 4 <= grade <= high + 4:
        return 0.4
    return 0.0

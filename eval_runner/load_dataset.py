"""Load JSONL evaluation dataset into structured objects."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_dataset(path: Path, max_items: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Load evaluation dataset from JSONL.
    
    Args:
        path: Path to .jsonl file
        max_items: If set, load only first max_items lines
        
    Returns:
        List of record dicts (one per line)
    """
    records = []
    with open(path, "r") as f:
        for i, line in enumerate(f):
            if max_items is not None and i >= max_items:
                break
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skip line {i+1}: {e}")
    logger.info(f"Loaded {len(records)} records from {path}")
    return records


def filter_by_course(records: List[Dict[str, Any]], course_id: str) -> List[Dict[str, Any]]:
    """Return records whose course_id matches."""
    return [r for r in records if r.get("course_id") == course_id]


def get_profiles_to_test(record: Dict[str, Any]) -> List[str]:
    """Return list of profile keys to test (e.g. undergrad, masters, phd)."""
    targets = record.get("personalization_targets") or {}
    if targets:
        return list(targets.keys())
    return ["undergrad", "masters", "phd"]

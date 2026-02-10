"""
Per-question scores -> per-variant, per-category, overall aggregates (mean, std).
"""

import logging
from typing import Dict, Any, List
from collections import defaultdict

logger = logging.getLogger(__name__)


def aggregate_scores(scored_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    scored_runs: list of {eval_id, category, variant, profile, scores: {metric: value}, ...}
    Returns:
      - per_variant: {variant: {metric: {mean, std, n}}}
      - per_category: {category: {metric: {mean, std, n}}}
      - per_variant_category: {(variant, category): {metric: {mean, std, n}}}
      - overall: {metric: {mean, std, n}}
      - deltas: {variant: {metric: full_system_mean - variant_mean}} (full_system as baseline)
    """
    per_variant: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_category: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_variant_category: Dict[tuple, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    all_scores: Dict[str, List[float]] = defaultdict(list)

    for run in scored_runs:
        variant = run.get("variant", "")
        category = run.get("category", "")
        scores = run.get("scores") or {}
        for metric, value in scores.items():
            if value is None:
                continue
            try:
                v = float(value)
            except (TypeError, ValueError):
                continue
            per_variant[variant][metric].append(v)
            per_category[category][metric].append(v)
            per_variant_category[(variant, category)][metric].append(v)
            all_scores[metric].append(v)

    def stats(lst: List[float]) -> Dict[str, Any]:
        if not lst:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        import statistics
        return {"mean": statistics.mean(lst), "std": statistics.stdev(lst) if len(lst) > 1 else 0.0, "n": len(lst)}

    out = {
        "per_variant": {v: {m: stats(per_variant[v][m]) for m in per_variant[v]} for v in per_variant},
        "per_category": {c: {m: stats(per_category[c][m]) for m in per_category[c]} for c in per_category},
        "per_variant_category": {k: {m: stats(per_variant_category[k][m]) for m in per_variant_category[k]} for k in per_variant_category},
        "overall": {m: stats(all_scores[m]) for m in all_scores},
    }

    # Deltas: full_system - variant for each metric
    full_means = out["per_variant"].get("full_system", {})
    deltas = {}
    for variant in out["per_variant"]:
        if variant == "full_system":
            continue
        deltas[variant] = {}
        for metric in out["per_variant"].get(variant, {}):
            fm = full_means.get(metric, {}).get("mean", 0.0)
            vm = out["per_variant"][variant].get(metric, {}).get("mean", 0.0)
            deltas[variant][metric] = fm - vm
    out["deltas"] = deltas

    return out

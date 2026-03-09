"""Aggregate evaluation scores for reporting."""

import numpy as np
from collections import defaultdict


def aggregate_scores(scored_runs: list) -> dict:
    """Aggregate scored runs into per-variant, per-category, and overall summaries.

    Returns dict with keys:
        per_variant: {variant -> {metric -> {mean, std, n}}}
        per_category: {category -> {metric -> {mean, std, n}}}
        per_variant_category: {(variant, category) -> {metric -> {mean, std, n}}}
        per_course: {course_id -> {metric -> {mean, std, n}}}
        per_profile: {profile -> {metric -> {mean, std, n}}}
        overall: {metric -> {mean, std, n}}
        deltas: {variant -> {metric -> delta_vs_full_system}}
    """
    # Collect scores by various groupings
    by_variant = defaultdict(lambda: defaultdict(list))
    by_category = defaultdict(lambda: defaultdict(list))
    by_variant_category = defaultdict(lambda: defaultdict(list))
    by_course = defaultdict(lambda: defaultdict(list))
    by_profile = defaultdict(lambda: defaultdict(list))
    overall = defaultdict(list)

    for run in scored_runs:
        variant = run.get("variant", "unknown")
        category = run.get("category", "unknown")
        course = run.get("course_id", "unknown")
        profile = run.get("profile", "unknown")
        scores = run.get("scores", {})

        # Use averaged scores if available (dual-judge), otherwise raw
        metric_scores = scores.get("averaged", scores)

        for metric, value in metric_scores.items():
            if value is None:
                continue
            try:
                val = float(value)
            except (TypeError, ValueError):
                continue

            by_variant[variant][metric].append(val)
            by_category[category][metric].append(val)
            by_variant_category[(variant, category)][metric].append(val)
            by_course[course][metric].append(val)
            by_profile[profile][metric].append(val)
            overall[metric].append(val)

    def summarize(grouped):
        result = {}
        for key, metrics in grouped.items():
            result[key] = {}
            for metric, values in metrics.items():
                arr = np.array(values)
                result[key][metric] = {
                    "mean": round(float(arr.mean()), 4),
                    "std": round(float(arr.std()), 4),
                    "n": len(values),
                }
        return result

    per_variant = summarize(by_variant)
    per_category = summarize(by_category)
    per_course = summarize(by_course)
    per_profile = summarize(by_profile)

    # variant_category uses tuple keys — convert to string keys for JSON
    vc_summarized = summarize(by_variant_category)
    per_variant_category = {}
    for (v, c), metrics in vc_summarized.items():
        per_variant_category[f"{v}|{c}"] = metrics

    overall_summary = {}
    for metric, values in overall.items():
        arr = np.array(values)
        overall_summary[metric] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "n": len(values),
        }

    # Compute deltas vs full_system
    deltas = {}
    full = per_variant.get("full_system", {})
    for variant, metrics in per_variant.items():
        if variant == "full_system":
            continue
        deltas[variant] = {}
        for metric in metrics:
            full_mean = full.get(metric, {}).get("mean")
            var_mean = metrics[metric].get("mean")
            if full_mean is not None and var_mean is not None:
                deltas[variant][metric] = round(full_mean - var_mean, 4)

    return {
        "per_variant": per_variant,
        "per_category": per_category,
        "per_variant_category": per_variant_category,
        "per_course": per_course,
        "per_profile": per_profile,
        "overall": overall_summary,
        "deltas": deltas,
    }

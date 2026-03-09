"""Compute weighted composite scores and statistical tests for PRISM evaluation.

Addresses evaluation fairness: baseline skips RAG metrics (no retrieval context),
inflating its apparent quality. This module penalizes missing metrics and computes
proper composite scores.
"""

import json
import numpy as np
from collections import defaultdict
from scipy import stats
from pathlib import Path


# ── Metric Groups with Weights ──────────────────────────────────────────────

# Group 1: Orchestration — Does the system route and use tools correctly?
ORCHESTRATION_METRICS = {
    "routing_accuracy": 0.35,
    "tool_correctness": 0.35,
    "task_completion": 0.30,
}

# Group 2: RAG Quality — Does retrieval actually help?
RAG_METRICS = {
    "contextual_precision": 0.30,
    "contextual_recall": 0.30,
    "correctness": 0.40,
}

# Group 3: Response Quality — Is the answer good?
RESPONSE_METRICS = {
    "answer_relevancy": 0.30,
    "coherence": 0.25,
    "personalization_accuracy": 0.25,
    "readability": 0.20,
}

# Group 4: Safety — Is the answer safe?
SAFETY_METRICS = {
    "toxicity": 0.40,       # lower is better → will invert
    "bias": 0.40,           # lower is better → will invert
    "refusal_correctness": 0.20,
}

# Group 5: Category-specific
VAGUE_METRICS = {
    "clarification_quality": 0.50,
    "routing_accuracy": 0.25,
    "task_completion": 0.25,
}

OOS_METRICS = {
    "refusal_correctness": 0.50,
    "routing_accuracy": 0.25,
    "task_completion": 0.25,
}

# Lower-is-better metrics (invert for composite: score = 1 - value)
INVERT_METRICS = {"toxicity", "bias", "hallucination"}

# Drop unreliable metrics (inter-judge r < 0.3)
UNRELIABLE_METRICS = {"faithfulness"}  # r = -0.12 between judges

# Overall composite weights for the 4 groups
GROUP_WEIGHTS = {
    "orchestration": 0.30,
    "rag_quality": 0.20,
    "response_quality": 0.35,
    "safety": 0.15,
}


def get_metric_value(scores, metric):
    """Get metric value, handling inversions and missing values."""
    val = scores.get(metric)
    if val is None:
        return None
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if metric in INVERT_METRICS:
        v = 1.0 - v
    return v


def compute_group_score(scores, metric_weights):
    """Compute weighted score for a metric group. Missing metrics get 0."""
    total_weight = sum(metric_weights.values())
    weighted_sum = 0.0
    available_weight = 0.0

    for metric, weight in metric_weights.items():
        if metric in UNRELIABLE_METRICS:
            continue
        val = get_metric_value(scores, metric)
        if val is not None:
            weighted_sum += val * weight
            available_weight += weight

    if available_weight == 0:
        return None

    # Penalize missing metrics: if only 60% of metrics available,
    # scale score by coverage fraction
    coverage = available_weight / total_weight
    raw_score = weighted_sum / available_weight
    penalized_score = raw_score * coverage

    return round(penalized_score, 4)


def compute_composite(scores, category):
    """Compute overall composite score for a run."""
    if category in ("vague",):
        # Vague: use vague-specific metrics
        return compute_group_score(scores, VAGUE_METRICS)
    elif category in ("out_of_scope",):
        return compute_group_score(scores, OOS_METRICS)

    # Standard categories: weighted combination of groups
    groups = {
        "orchestration": compute_group_score(scores, ORCHESTRATION_METRICS),
        "rag_quality": compute_group_score(scores, RAG_METRICS),
        "response_quality": compute_group_score(scores, RESPONSE_METRICS),
        "safety": compute_group_score(scores, SAFETY_METRICS),
    }

    weighted_sum = 0.0
    total_weight = 0.0
    for group, score in groups.items():
        if score is not None:
            w = GROUP_WEIGHTS[group]
            weighted_sum += score * w
            total_weight += w

    if total_weight == 0:
        return None

    return round(weighted_sum / total_weight, 4), groups


def analyze_runs(scored_runs):
    """Full analysis: composite scores, group scores, significance tests."""

    # Organize by (eval_id, profile) -> variant -> scores
    by_question = defaultdict(dict)  # (eval_id, profile) -> {variant: {scores, composite, ...}}

    for run in scored_runs:
        eid = run["eval_id"]
        variant = run["variant"]
        profile = run["profile"]
        category = run.get("category", "")
        scores = run.get("scores", {}).get("averaged", {})

        key = (eid, profile)

        result = compute_composite(scores, category)
        if isinstance(result, tuple):
            composite, groups = result
        else:
            composite = result
            groups = {}

        by_question[key][variant] = {
            "scores": scores,
            "composite": composite,
            "groups": groups,
            "category": category,
        }

    # ── Per-variant composite scores ──
    variant_composites = defaultdict(list)
    variant_groups = defaultdict(lambda: defaultdict(list))

    for qkey, variants in by_question.items():
        for variant, data in variants.items():
            if data["composite"] is not None:
                variant_composites[variant].append(data["composite"])
            for group, score in data.get("groups", {}).items():
                if score is not None:
                    variant_groups[variant][group].append(score)

    print("\n" + "=" * 80)
    print("COMPOSITE SCORES BY VARIANT (penalizes missing metrics)")
    print("=" * 80)

    composite_summary = {}
    for variant in sorted(variant_composites.keys()):
        scores = variant_composites[variant]
        mean = np.mean(scores)
        std = np.std(scores)
        composite_summary[variant] = {"mean": round(mean, 4), "std": round(std, 4), "n": len(scores)}
        print(f"  {variant:25s}  {mean:.4f} ± {std:.4f}  (n={len(scores)})")

    print("\n" + "=" * 80)
    print("GROUP SCORES BY VARIANT")
    print("=" * 80)

    group_summary = {}
    for variant in sorted(variant_groups.keys()):
        group_summary[variant] = {}
        parts = []
        for group in ["orchestration", "rag_quality", "response_quality", "safety"]:
            vals = variant_groups[variant].get(group, [])
            if vals:
                m = np.mean(vals)
                group_summary[variant][group] = round(m, 4)
                parts.append(f"{group}={m:.3f}")
            else:
                parts.append(f"{group}=N/A")
        print(f"  {variant:25s}  {' | '.join(parts)}")

    # ── Paired significance tests: full_system vs each variant ──
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE: full_system vs each variant (paired t-test)")
    print("=" * 80)

    significance = {}
    for variant in sorted(variant_composites.keys()):
        if variant == "full_system":
            continue

        # Collect paired observations
        full_vals = []
        other_vals = []
        for qkey, variants_data in by_question.items():
            if "full_system" in variants_data and variant in variants_data:
                fc = variants_data["full_system"]["composite"]
                oc = variants_data[variant]["composite"]
                if fc is not None and oc is not None:
                    full_vals.append(fc)
                    other_vals.append(oc)

        if len(full_vals) < 10:
            print(f"  {variant:25s}  too few paired observations ({len(full_vals)})")
            continue

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(full_vals, other_vals)
        # Effect size (Cohen's d for paired samples)
        diffs = np.array(full_vals) - np.array(other_vals)
        cohens_d = np.mean(diffs) / np.std(diffs) if np.std(diffs) > 0 else 0
        # Wilcoxon signed-rank (non-parametric)
        try:
            w_stat, w_p = stats.wilcoxon(diffs)
        except ValueError:
            w_stat, w_p = 0, 1.0

        delta = np.mean(full_vals) - np.mean(other_vals)
        sig_label = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

        significance[variant] = {
            "delta": round(delta, 4),
            "t_stat": round(t_stat, 4),
            "p_value": round(p_value, 6),
            "cohens_d": round(cohens_d, 4),
            "wilcoxon_p": round(w_p, 6),
            "n_pairs": len(full_vals),
            "significant": p_value < 0.05,
        }

        print(f"  full_system vs {variant:20s}  Δ={delta:+.4f}  t={t_stat:7.3f}  "
              f"p={p_value:.6f} {sig_label}  d={cohens_d:.3f}  n={len(full_vals)}")

    # ── Per-metric significance (full_system vs baseline) ──
    print("\n" + "=" * 80)
    print("PER-METRIC SIGNIFICANCE: full_system vs baseline")
    print("=" * 80)

    metric_significance = {}
    all_metrics = set()
    for qkey, variants_data in by_question.items():
        for v, d in variants_data.items():
            all_metrics.update(d["scores"].keys())

    for metric in sorted(all_metrics):
        if metric in UNRELIABLE_METRICS:
            continue
        full_vals = []
        base_vals = []
        for qkey, variants_data in by_question.items():
            if "full_system" in variants_data and "baseline" in variants_data:
                fv = variants_data["full_system"]["scores"].get(metric)
                bv = variants_data["baseline"]["scores"].get(metric)
                if fv is not None and bv is not None:
                    try:
                        full_vals.append(float(fv))
                        base_vals.append(float(bv))
                    except (TypeError, ValueError):
                        continue

        if len(full_vals) < 10:
            print(f"  {metric:30s}  n={len(full_vals):4d}  (too few pairs)")
            continue

        t_stat, p_value = stats.ttest_rel(full_vals, base_vals)
        delta = np.mean(full_vals) - np.mean(base_vals)
        sig_label = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

        metric_significance[metric] = {
            "full_mean": round(np.mean(full_vals), 4),
            "baseline_mean": round(np.mean(base_vals), 4),
            "delta": round(delta, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < 0.05,
            "n": len(full_vals),
        }

        winner = "PRISM" if delta > 0 else "BASE"
        print(f"  {metric:30s}  PRISM={np.mean(full_vals):.4f}  BASE={np.mean(base_vals):.4f}  "
              f"Δ={delta:+.4f}  p={p_value:.6f} {sig_label}  [{winner}]")

    # ── Category-specific composite ──
    print("\n" + "=" * 80)
    print("COMPOSITE BY CATEGORY × VARIANT")
    print("=" * 80)

    cat_variant = defaultdict(lambda: defaultdict(list))
    for qkey, variants_data in by_question.items():
        for variant, data in variants_data.items():
            cat = data["category"]
            if data["composite"] is not None:
                cat_variant[cat][variant].append(data["composite"])

    category_results = {}
    for cat in sorted(cat_variant.keys()):
        print(f"\n  {cat}:")
        category_results[cat] = {}
        for variant in sorted(cat_variant[cat].keys()):
            vals = cat_variant[cat][variant]
            m = np.mean(vals)
            category_results[cat][variant] = round(m, 4)
            print(f"    {variant:25s}  {m:.4f} (n={len(vals)})")

    return {
        "composite_summary": composite_summary,
        "group_summary": group_summary,
        "significance_vs_variants": significance,
        "metric_significance_vs_baseline": metric_significance,
        "category_composites": category_results,
    }


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from prism_eval.config import RAW_DIR, TABLES_DIR

    # Load merged scored runs
    scored_runs = []
    with open(RAW_DIR / "scored_runs.jsonl") as f:
        for line in f:
            scored_runs.append(json.loads(line))

    print(f"Loaded {len(scored_runs)} scored runs\n")

    results = analyze_runs(scored_runs)

    # Save results
    out_path = TABLES_DIR / "composite_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

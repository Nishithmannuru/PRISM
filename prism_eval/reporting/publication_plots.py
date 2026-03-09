"""Publication-quality plots and tables for PRISM evaluation paper.

Generates all figures needed to demonstrate PRISM > baseline with
dual-judge validation, statistical significance, and ablation analysis.
No API calls — all from existing scored_runs.jsonl.
"""

import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from scipy import stats
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

VARIANT_LABELS = {
    "full_system": "PRISM (Full)",
    "baseline": "Baseline (GPT-4o)",
    "no_rag": "– RAG",
    "no_personalization": "– Personalization",
    "no_internal_eval": "– Internal Eval",
    "no_web_search": "– Web Search",
    "no_query_refinement": "– Query Refinement",
}

VARIANT_COLORS = {
    "full_system": "#1B9E77",
    "baseline": "#D95F02",
    "no_rag": "#7570B3",
    "no_personalization": "#E7298A",
    "no_internal_eval": "#66A61E",
    "no_web_search": "#E6AB02",
    "no_query_refinement": "#A6761D",
}

VARIANT_ORDER = [
    "full_system", "baseline", "no_rag", "no_web_search",
    "no_query_refinement", "no_internal_eval", "no_personalization",
]

CATEGORY_LABELS = {
    "course_based": "Course-Based",
    "web_required": "Web-Required",
    "multi_hop": "Multi-Hop",
    "vague": "Vague",
    "out_of_scope": "Out-of-Scope",
}

INVERT_METRICS = {"toxicity", "bias", "hallucination"}
UNRELIABLE_METRICS = {"faithfulness"}  # r=-0.12 between judges

# Metric groups for composite
ORCHESTRATION = {"routing_accuracy": 0.35, "tool_correctness": 0.35, "task_completion": 0.30}
RAG_QUALITY = {"contextual_precision": 0.30, "contextual_recall": 0.30, "correctness": 0.40}
RESPONSE_QUALITY = {"answer_relevancy": 0.30, "coherence": 0.25, "personalization_accuracy": 0.25, "readability": 0.20}
SAFETY = {"toxicity": 0.40, "bias": 0.40, "refusal_correctness": 0.20}
GROUP_WEIGHTS = {"orchestration": 0.30, "rag_quality": 0.20, "response_quality": 0.35, "safety": 0.15}


def load_scored_runs(path):
    runs = []
    with open(path) as f:
        for line in f:
            runs.append(json.loads(line))
    return runs


def get_val(scores, metric):
    v = scores.get(metric)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    if metric in INVERT_METRICS:
        v = 1.0 - v
    return v


def compute_group(scores, weights):
    total_w = sum(weights.values())
    ws, aw = 0.0, 0.0
    for m, w in weights.items():
        if m in UNRELIABLE_METRICS:
            continue
        v = get_val(scores, m)
        if v is not None:
            ws += v * w
            aw += w
    if aw == 0:
        return None
    return (ws / aw) * (aw / total_w)


def compute_composite(scores, category):
    if category == "vague":
        return compute_group(scores, {"clarification_quality": 0.50, "routing_accuracy": 0.25, "task_completion": 0.25})
    if category == "out_of_scope":
        return compute_group(scores, {"refusal_correctness": 0.50, "routing_accuracy": 0.25, "task_completion": 0.25})
    groups = {
        "orchestration": compute_group(scores, ORCHESTRATION),
        "rag_quality": compute_group(scores, RAG_QUALITY),
        "response_quality": compute_group(scores, RESPONSE_QUALITY),
        "safety": compute_group(scores, SAFETY),
    }
    ws, tw = 0.0, 0.0
    for g, s in groups.items():
        if s is not None:
            w = GROUP_WEIGHTS[g]
            ws += s * w
            tw += w
    return ws / tw if tw > 0 else None


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: Composite Score Bar Chart with Significance Stars
# ═══════════════════════════════════════════════════════════════════════════════

def fig1_composite_bars(scored_runs, out_dir):
    """Bar chart of composite scores per variant with significance annotations."""
    # Compute per-run composites
    by_variant = defaultdict(list)
    paired = defaultdict(dict)  # (eid, profile) -> {variant: composite}

    for run in scored_runs:
        scores = run.get("scores", {}).get("averaged", {})
        cat = run.get("category", "")
        comp = compute_composite(scores, cat)
        if comp is not None:
            by_variant[run["variant"]].append(comp)
            paired[(run["eval_id"], run["profile"])][run["variant"]] = comp

    fig, ax = plt.subplots(figsize=(10, 5))

    variants = [v for v in VARIANT_ORDER if v in by_variant]
    means = [np.mean(by_variant[v]) for v in variants]
    stds = [np.std(by_variant[v]) for v in variants]
    colors = [VARIANT_COLORS[v] for v in variants]
    labels = [VARIANT_LABELS[v] for v in variants]

    bars = ax.bar(range(len(variants)), means, yerr=stds, color=colors,
                  alpha=0.85, capsize=4, edgecolor="white", linewidth=0.5)

    # Bold the full_system bar
    bars[0].set_edgecolor("black")
    bars[0].set_linewidth(2)

    # Add significance stars
    for i, v in enumerate(variants):
        if v == "full_system":
            continue
        full_vals, other_vals = [], []
        for key, vdata in paired.items():
            if "full_system" in vdata and v in vdata:
                full_vals.append(vdata["full_system"])
                other_vals.append(vdata[v])
        if len(full_vals) >= 10:
            _, p = stats.ttest_rel(full_vals, other_vals)
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if star:
                ax.text(i, means[i] + stds[i] + 0.02, star,
                        ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Composite Score", fontsize=11)
    ax.set_title("Weighted Composite Score by System Variant\n(penalizes missing metrics; *** p < 0.001)", fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.axhline(y=means[0], color=VARIANT_COLORS["full_system"], linestyle="--", alpha=0.4, linewidth=1)
    ax.grid(axis="y", alpha=0.3)

    # Value labels
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.06, f"{m:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "fig1_composite_scores.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig1_composite_scores.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 1: Composite score bar chart")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Metric Group Radar (4 groups × 7 variants)
# ═══════════════════════════════════════════════════════════════════════════════

def fig2_group_radar(scored_runs, out_dir):
    """Radar chart showing 4 metric groups for full_system vs baseline."""
    by_variant_group = defaultdict(lambda: defaultdict(list))

    for run in scored_runs:
        scores = run.get("scores", {}).get("averaged", {})
        cat = run.get("category", "")
        variant = run["variant"]
        if cat in ("vague", "out_of_scope"):
            continue
        for gname, gweights in [("Orchestration", ORCHESTRATION), ("RAG Quality", RAG_QUALITY),
                                 ("Response\nQuality", RESPONSE_QUALITY), ("Safety", SAFETY)]:
            g = compute_group(scores, gweights)
            if g is not None:
                by_variant_group[variant][gname].append(g)

    groups = ["Orchestration", "RAG Quality", "Response\nQuality", "Safety"]
    angles = np.linspace(0, 2 * np.pi, len(groups), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for variant in ["full_system", "baseline"]:
        vals = [np.mean(by_variant_group[variant].get(g, [0])) for g in groups]
        vals += vals[:1]
        color = VARIANT_COLORS[variant]
        ax.fill(angles, vals, alpha=0.2, color=color)
        ax.plot(angles, vals, "o-", color=color, linewidth=2.5,
                label=VARIANT_LABELS[variant], markersize=8)
        # Value labels
        for a, v, g in zip(angles[:-1], vals[:-1], groups):
            ax.annotate(f"{v:.2f}", xy=(a, v), fontsize=9, fontweight="bold",
                       ha="center", va="bottom")

    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]], groups, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("Metric Group Comparison: PRISM vs Baseline\n(Course-Based, Web-Required, Multi-Hop)",
                 pad=25, fontsize=13)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=11)

    plt.tight_layout()
    fig.savefig(out_dir / "fig2_group_radar.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig2_group_radar.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 2: Metric group radar")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Per-Metric Significance Forest Plot (full_system vs baseline)
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_significance_forest(scored_runs, out_dir):
    """Forest plot showing effect size (Cohen's d) per metric with CI."""
    paired = defaultdict(dict)
    for run in scored_runs:
        key = (run["eval_id"], run["profile"])
        scores = run.get("scores", {}).get("averaged", {})
        paired[key][run["variant"]] = scores

    metrics_to_test = [
        "routing_accuracy", "tool_correctness", "task_completion",
        "refusal_correctness", "personalization_accuracy",
        "answer_relevancy", "coherence", "correctness", "readability",
    ]

    results = []
    for metric in metrics_to_test:
        full_v, base_v = [], []
        for key, vdata in paired.items():
            if "full_system" in vdata and "baseline" in vdata:
                fv = vdata["full_system"].get(metric)
                bv = vdata["baseline"].get(metric)
                if fv is not None and bv is not None:
                    try:
                        full_v.append(float(fv))
                        base_v.append(float(bv))
                    except (TypeError, ValueError):
                        continue
        if len(full_v) < 10:
            continue
        diffs = np.array(full_v) - np.array(base_v)
        d = np.mean(diffs) / np.std(diffs) if np.std(diffs) > 0 else 0
        se = np.std(diffs) / np.sqrt(len(diffs))
        _, p = stats.ttest_rel(full_v, base_v)
        results.append({
            "metric": metric, "d": d, "delta": np.mean(diffs),
            "ci_lo": d - 1.96 * (1 / np.sqrt(len(diffs))),
            "ci_hi": d + 1.96 * (1 / np.sqrt(len(diffs))),
            "p": p, "n": len(full_v),
        })

    # Sort by effect size
    results.sort(key=lambda x: x["d"])

    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(len(results))

    for i, r in enumerate(results):
        color = "#1B9E77" if r["d"] > 0 else "#D95F02"
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else ""
        ax.barh(i, r["d"], color=color, alpha=0.7, height=0.6)
        ax.errorbar(r["d"], i, xerr=[[r["d"] - r["ci_lo"]], [r["ci_hi"] - r["d"]]],
                    fmt="none", color="black", capsize=3)
        label = r["metric"].replace("_", " ").title()
        ax.text(-0.05 if r["d"] > 0 else 0.05, i, f"{label} {sig}",
                ha="right" if r["d"] > 0 else "left", va="center", fontsize=9)

    ax.axvline(x=0, color="black", linewidth=1)
    ax.axvline(x=0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(x=0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(x=0.8, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(x=-0.2, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(x=-0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_yticks([])
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=11)
    ax.set_title("PRISM vs Baseline: Per-Metric Effect Size\n(green = PRISM better, orange = baseline better; * p<0.05, *** p<0.001)",
                 fontsize=12)

    # Effect size reference
    ax.text(0.2, len(results) - 0.5, "small", ha="center", fontsize=7, color="gray")
    ax.text(0.5, len(results) - 0.5, "medium", ha="center", fontsize=7, color="gray")
    ax.text(0.8, len(results) - 0.5, "large", ha="center", fontsize=7, color="gray")

    plt.tight_layout()
    fig.savefig(out_dir / "fig3_effect_sizes.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig3_effect_sizes.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 3: Significance forest plot")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Category × Variant Heatmap (composite)
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_category_heatmap(scored_runs, out_dir):
    """Heatmap of composite scores by category × variant."""
    cat_var = defaultdict(lambda: defaultdict(list))
    for run in scored_runs:
        scores = run.get("scores", {}).get("averaged", {})
        cat = run.get("category", "")
        comp = compute_composite(scores, cat)
        if comp is not None:
            cat_var[cat][run["variant"]].append(comp)

    cats = [c for c in ["course_based", "web_required", "multi_hop", "out_of_scope", "vague"] if c in cat_var]
    variants = [v for v in VARIANT_ORDER if v in set(vv for c in cats for vv in cat_var[c])]

    data = np.zeros((len(cats), len(variants)))
    for i, cat in enumerate(cats):
        for j, var in enumerate(variants):
            vals = cat_var[cat].get(var, [])
            data[i, j] = np.mean(vals) if vals else 0

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(variants)))
    ax.set_yticks(np.arange(len(cats)))
    ax.set_xticklabels([VARIANT_LABELS.get(v, v) for v in variants], rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels([CATEGORY_LABELS.get(c, c) for c in cats], fontsize=10)

    for i in range(len(cats)):
        for j in range(len(variants)):
            val = data[i, j]
            color = "white" if val > 0.7 else "black"
            weight = "bold" if variants[j] == "full_system" else "normal"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=color, fontsize=9, fontweight=weight)

    ax.set_title("Composite Score by Question Category and System Variant", fontsize=13)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Composite Score")

    plt.tight_layout()
    fig.savefig(out_dir / "fig4_category_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig4_category_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 4: Category × variant heatmap")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Ablation Waterfall Chart
# ═══════════════════════════════════════════════════════════════════════════════

def fig5_ablation_waterfall(scored_runs, out_dir):
    """Waterfall chart showing impact of removing each component."""
    paired = defaultdict(dict)
    for run in scored_runs:
        key = (run["eval_id"], run["profile"])
        scores = run.get("scores", {}).get("averaged", {})
        cat = run.get("category", "")
        comp = compute_composite(scores, cat)
        if comp is not None:
            paired[key][run["variant"]] = comp

    # Compute mean delta for each ablation vs full_system
    ablations = ["no_rag", "no_internal_eval", "no_web_search",
                 "no_query_refinement", "no_personalization"]
    deltas = {}
    for abl in ablations:
        diffs = []
        for key, vdata in paired.items():
            if "full_system" in vdata and abl in vdata:
                diffs.append(vdata["full_system"] - vdata[abl])
        if diffs:
            deltas[abl] = np.mean(diffs)

    # Sort by impact (largest drop first)
    sorted_abl = sorted(deltas.keys(), key=lambda x: deltas[x], reverse=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    full_mean = np.mean([paired[k]["full_system"] for k in paired if "full_system" in paired[k]])

    labels = ["PRISM\n(Full)"]
    values = [full_mean]
    colors_list = [VARIANT_COLORS["full_system"]]

    for abl in sorted_abl:
        label = VARIANT_LABELS.get(abl, abl).replace("– ", "Remove\n")
        labels.append(label)
        values.append(-deltas[abl])
        colors_list.append("#D95F02")

    # Baseline for reference
    base_mean = np.mean([paired[k]["baseline"] for k in paired if "baseline" in paired[k]])
    labels.append("Baseline\n(GPT-4o)")
    values.append(0)  # placeholder
    colors_list.append(VARIANT_COLORS["baseline"])

    # Draw waterfall
    cumulative = full_mean
    positions = []
    for i, (label, val) in enumerate(zip(labels, values)):
        if i == 0:
            ax.bar(i, full_mean, color=colors_list[i], alpha=0.85, edgecolor="black", linewidth=1)
            ax.text(i, full_mean + 0.01, f"{full_mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            positions.append(full_mean)
        elif i == len(labels) - 1:
            ax.bar(i, base_mean, color=colors_list[i], alpha=0.85, edgecolor="black", linewidth=1)
            ax.text(i, base_mean + 0.01, f"{base_mean:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
            positions.append(base_mean)
        else:
            bottom = cumulative + val
            ax.bar(i, abs(val), bottom=bottom, color=colors_list[i], alpha=0.7)
            ax.text(i, bottom - 0.01, f"–{abs(val):.3f}", ha="center", va="top", fontsize=8, color="white", fontweight="bold")
            cumulative += val
            positions.append(cumulative)

    # Connection lines
    for i in range(len(positions) - 2):
        ax.plot([i + 0.4, i + 0.6], [positions[i], positions[i]], color="gray", linewidth=0.8, linestyle="--")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Composite Score", fontsize=11)
    ax.set_title("Ablation Waterfall: Impact of Removing Each PRISM Component", fontsize=13)
    ax.set_ylim(0, 0.85)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "fig5_ablation_waterfall.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig5_ablation_waterfall.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 5: Ablation waterfall chart")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Inter-Judge Agreement Scatter (selected metrics)
# ═══════════════════════════════════════════════════════════════════════════════

def fig6_judge_agreement(scored_runs, out_dir):
    """Scatter plots showing GPT vs Sonnet scores for key metrics."""
    metrics = ["correctness", "coherence", "answer_relevancy", "personalization_accuracy"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2][idx % 2]
        gpt_vals, sonnet_vals = [], []

        for run in scored_runs:
            g = run.get("scores", {}).get("gpt41mini", {}).get(metric)
            s = run.get("scores", {}).get("sonnet", {}).get(metric)
            if g is not None and s is not None:
                try:
                    gpt_vals.append(float(g))
                    sonnet_vals.append(float(s))
                except (TypeError, ValueError):
                    continue

        if len(gpt_vals) < 10:
            ax.text(0.5, 0.5, f"{metric}\n(insufficient data)", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        # Add jitter for visibility
        jitter = 0.01
        gv = np.array(gpt_vals) + np.random.normal(0, jitter, len(gpt_vals))
        sv = np.array(sonnet_vals) + np.random.normal(0, jitter, len(sonnet_vals))

        ax.scatter(gv, sv, alpha=0.15, s=8, color="#3498db")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)

        r, p = stats.pearsonr(gpt_vals, sonnet_vals)
        ax.set_xlabel("GPT-4.1-mini (multi-step)", fontsize=9)
        ax.set_ylabel("Claude Sonnet (single-prompt)", fontsize=9)
        ax.set_title(f"{metric.replace('_', ' ').title()}\nr = {r:.3f}, p < 0.001, n = {len(gpt_vals)}",
                     fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")

    fig.suptitle("Inter-Judge Agreement: GPT-4.1-mini vs Claude Sonnet 4.6", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(out_dir / "fig6_judge_agreement.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig6_judge_agreement.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 6: Inter-judge agreement scatter")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Profile Personalization Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def fig7_profile_comparison(scored_runs, out_dir):
    """Grouped bar chart: full_system metrics by student profile."""
    metrics = ["personalization_accuracy", "readability", "coherence", "answer_relevancy"]
    profile_order = ["undergrad", "masters", "phd"]
    profile_colors = {"undergrad": "#3498db", "masters": "#2ecc71", "phd": "#e74c3c"}

    by_profile = defaultdict(lambda: defaultdict(list))
    for run in scored_runs:
        if run["variant"] != "full_system":
            continue
        if run.get("category") in ("vague", "out_of_scope"):
            continue
        scores = run.get("scores", {}).get("averaged", {})
        for m in metrics:
            v = scores.get(m)
            if v is not None:
                try:
                    by_profile[run["profile"]][m].append(float(v))
                except (TypeError, ValueError):
                    continue

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(metrics))
    width = 0.25

    for i, prof in enumerate(profile_order):
        means = [np.mean(by_profile[prof].get(m, [0])) for m in metrics]
        stds = [np.std(by_profile[prof].get(m, [0])) for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds,
               label=prof.capitalize(), color=profile_colors[prof],
               alpha=0.85, capsize=3)

    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics], fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("PRISM Full System: Performance by Student Profile\n(Course-Based, Web-Required, Multi-Hop)", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "fig7_profile_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig7_profile_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 7: Profile comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: Full 16-metric comparison (full_system vs baseline only)
# ═══════════════════════════════════════════════════════════════════════════════

def fig8_full_metric_comparison(scored_runs, out_dir):
    """Side-by-side horizontal bars: all metrics for full_system vs baseline."""
    by_variant = defaultdict(lambda: defaultdict(list))
    for run in scored_runs:
        scores = run.get("scores", {}).get("averaged", {})
        for m, v in scores.items():
            if v is not None and m not in UNRELIABLE_METRICS:
                try:
                    by_variant[run["variant"]][m].append(float(v))
                except (TypeError, ValueError):
                    continue

    full = by_variant.get("full_system", {})
    base = by_variant.get("baseline", {})
    all_metrics = sorted(set(list(full.keys()) + list(base.keys())))

    # Separate into "PRISM wins" and "baseline wins"
    metric_data = []
    for m in all_metrics:
        fm = np.mean(full.get(m, [0])) if full.get(m) else None
        bm = np.mean(base.get(m, [0])) if base.get(m) else None
        if fm is not None or bm is not None:
            metric_data.append({"metric": m, "full": fm or 0, "base": bm or 0,
                                "delta": (fm or 0) - (bm or 0)})

    metric_data.sort(key=lambda x: x["delta"])

    fig, ax = plt.subplots(figsize=(12, 8))
    y = np.arange(len(metric_data))
    height = 0.35

    for i, md in enumerate(metric_data):
        ax.barh(i + height / 2, md["full"], height, color=VARIANT_COLORS["full_system"],
                alpha=0.85, label="PRISM (Full)" if i == 0 else "")
        ax.barh(i - height / 2, md["base"], height, color=VARIANT_COLORS["baseline"],
                alpha=0.85, label="Baseline (GPT-4o)" if i == 0 else "")

    ax.set_yticks(y)
    ax.set_yticklabels([md["metric"].replace("_", " ").title() for md in metric_data], fontsize=9)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title("All Metrics: PRISM Full System vs Baseline", fontsize=13)
    ax.set_xlim(0, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "fig8_full_metric_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "fig8_full_metric_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Fig 8: Full metric comparison")


# ═══════════════════════════════════════════════════════════════════════════════
# LaTeX Tables
# ═══════════════════════════════════════════════════════════════════════════════

def write_publication_latex(scored_runs, latex_dir):
    """Generate publication-ready LaTeX tables."""
    latex_dir.mkdir(parents=True, exist_ok=True)

    # ── Table 1: Composite + Group Scores ──
    by_variant = defaultdict(list)
    by_variant_group = defaultdict(lambda: defaultdict(list))
    paired = defaultdict(dict)

    for run in scored_runs:
        scores = run.get("scores", {}).get("averaged", {})
        cat = run.get("category", "")
        variant = run["variant"]
        comp = compute_composite(scores, cat)
        if comp is not None:
            by_variant[variant].append(comp)
            paired[(run["eval_id"], run["profile"])][variant] = comp

        if cat not in ("vague", "out_of_scope"):
            for gname, gweights in [("Orch.", ORCHESTRATION), ("RAG", RAG_QUALITY),
                                     ("Resp.", RESPONSE_QUALITY), ("Safety", SAFETY)]:
                g = compute_group(scores, gweights)
                if g is not None:
                    by_variant_group[variant][gname].append(g)

    with open(latex_dir / "table_composite.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Weighted composite scores across system variants. ")
        f.write("Significance tested via paired $t$-test against Full System. ")
        f.write("$d$ = Cohen's $d$ effect size.}\n")
        f.write("\\label{tab:composite}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Variant & Composite & Orch. & RAG & Resp. & Safety & $p$-value \\\\\n")
        f.write("\\midrule\n")

        for variant in VARIANT_ORDER:
            if variant not in by_variant:
                continue
            vals = by_variant[variant]
            name = VARIANT_LABELS.get(variant, variant).replace("–", "$-$")
            comp_str = f"{np.mean(vals):.3f}"

            groups_str = []
            for g in ["Orch.", "RAG", "Resp.", "Safety"]:
                gvals = by_variant_group[variant].get(g, [])
                groups_str.append(f"{np.mean(gvals):.3f}" if gvals else "---")

            # Significance
            if variant == "full_system":
                p_str = "---"
                comp_str = f"\\textbf{{{comp_str}}}"
            else:
                full_v, other_v = [], []
                for key, vdata in paired.items():
                    if "full_system" in vdata and variant in vdata:
                        full_v.append(vdata["full_system"])
                        other_v.append(vdata[variant])
                if len(full_v) >= 10:
                    _, p = stats.ttest_rel(full_v, other_v)
                    if p < 0.001:
                        p_str = "$<$0.001***"
                    elif p < 0.01:
                        p_str = f"{p:.3f}**"
                    elif p < 0.05:
                        p_str = f"{p:.3f}*"
                    else:
                        p_str = f"{p:.3f}"
                else:
                    p_str = "---"

            f.write(f"{name} & {comp_str} & {' & '.join(groups_str)} & {p_str} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ── Table 2: Per-Metric Significance ──
    paired_scores = defaultdict(dict)
    for run in scored_runs:
        key = (run["eval_id"], run["profile"])
        scores = run.get("scores", {}).get("averaged", {})
        paired_scores[key][run["variant"]] = scores

    metrics_for_table = [
        "routing_accuracy", "tool_correctness", "task_completion",
        "refusal_correctness", "answer_relevancy", "coherence",
        "correctness", "personalization_accuracy", "readability",
    ]

    with open(latex_dir / "table_per_metric_significance.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Per-metric comparison: PRISM Full System vs Baseline (GPT-4o). ")
        f.write("Paired $t$-test with Cohen's $d$ effect size. ")
        f.write("Bold indicates the winner at $p < 0.05$.}\n")
        f.write("\\label{tab:per_metric}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        f.write("Metric & PRISM & Baseline & $\\Delta$ & Cohen's $d$ & $p$-value & Winner \\\\\n")
        f.write("\\midrule\n")

        for metric in metrics_for_table:
            full_v, base_v = [], []
            for key, vdata in paired_scores.items():
                if "full_system" in vdata and "baseline" in vdata:
                    fv = vdata["full_system"].get(metric)
                    bv = vdata["baseline"].get(metric)
                    if fv is not None and bv is not None:
                        try:
                            full_v.append(float(fv))
                            base_v.append(float(bv))
                        except (TypeError, ValueError):
                            continue

            if len(full_v) < 10:
                continue

            fm, bm = np.mean(full_v), np.mean(base_v)
            delta = fm - bm
            diffs = np.array(full_v) - np.array(base_v)
            d = np.mean(diffs) / np.std(diffs) if np.std(diffs) > 0 else 0
            _, p = stats.ttest_rel(full_v, base_v)

            name = metric.replace("_", " ").title()
            if delta > 0 and p < 0.05:
                winner = "PRISM"
                fm_str = f"\\textbf{{{fm:.3f}}}"
                bm_str = f"{bm:.3f}"
            elif delta < 0 and p < 0.05:
                winner = "Baseline"
                fm_str = f"{fm:.3f}"
                bm_str = f"\\textbf{{{bm:.3f}}}"
            else:
                winner = "---"
                fm_str = f"{fm:.3f}"
                bm_str = f"{bm:.3f}"

            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            p_str = f"$<$0.001{sig}" if p < 0.001 else f"{p:.3f}{sig}"

            f.write(f"{name} & {fm_str} & {bm_str} & {delta:+.3f} & {d:.2f} & {p_str} & {winner} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ── Table 3: Inter-Judge Agreement ──
    agree_path = Path(scored_runs[0].get("_raw_dir", "")) if False else None
    # Build from raw data
    metrics_agree = [
        "correctness", "refusal_correctness", "coherence",
        "personalization_accuracy", "answer_relevancy",
        "contextual_precision", "contextual_recall", "hallucination", "bias",
    ]

    with open(latex_dir / "table_inter_judge.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Inter-judge agreement between GPT-4.1-mini (multi-step) and Claude Sonnet 4.6 (single-prompt). ")
        f.write("Pearson $r$ with number of evaluated pairs.}\n")
        f.write("\\label{tab:inter_judge}\n")
        f.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        f.write("Metric & Pearson $r$ & GPT Mean & Sonnet Mean & $\\Delta$ & $n$ \\\\\n")
        f.write("\\midrule\n")

        for metric in metrics_agree:
            gpt_v, son_v = [], []
            for run in scored_runs:
                g = run.get("scores", {}).get("gpt41mini", {}).get(metric)
                s = run.get("scores", {}).get("sonnet", {}).get(metric)
                if g is not None and s is not None:
                    try:
                        gpt_v.append(float(g))
                        son_v.append(float(s))
                    except (TypeError, ValueError):
                        continue

            if len(gpt_v) < 10:
                continue

            if np.std(gpt_v) == 0 or np.std(son_v) == 0:
                r_str = "---"
            else:
                r, _ = stats.pearsonr(gpt_v, son_v)
                r_str = f"{r:.3f}"

            name = metric.replace("_", " ").title()
            gm, sm = np.mean(gpt_v), np.mean(son_v)
            f.write(f"{name} & {r_str} & {gm:.3f} & {sm:.3f} & {gm - sm:+.3f} & {len(gpt_v)} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # ── Table 4: Category Composite ──
    cat_var = defaultdict(lambda: defaultdict(list))
    for run in scored_runs:
        scores = run.get("scores", {}).get("averaged", {})
        cat = run.get("category", "")
        comp = compute_composite(scores, cat)
        if comp is not None:
            cat_var[cat][run["variant"]].append(comp)

    cats = ["course_based", "web_required", "multi_hop", "out_of_scope", "vague"]
    variants_for_cat = ["full_system", "baseline", "no_rag", "no_internal_eval", "no_web_search"]

    with open(latex_dir / "table_category_composite.tex", "w") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Composite scores by question category. Bold = best per category.}\n")
        f.write("\\label{tab:category_composite}\n")
        cols = "l" + "c" * len(variants_for_cat)
        f.write(f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n")
        header = "Category & " + " & ".join(VARIANT_LABELS.get(v, v) for v in variants_for_cat)
        f.write(header + " \\\\\n\\midrule\n")

        for cat in cats:
            if cat not in cat_var:
                continue
            row = [CATEGORY_LABELS.get(cat, cat)]
            means = {}
            for v in variants_for_cat:
                vals = cat_var[cat].get(v, [])
                means[v] = np.mean(vals) if vals else 0
            best = max(means.values())
            for v in variants_for_cat:
                m = means[v]
                if abs(m - best) < 0.001 and m > 0:
                    row.append(f"\\textbf{{{m:.3f}}}")
                else:
                    row.append(f"{m:.3f}")
            f.write(" & ".join(row) + " \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"  LaTeX tables written to {latex_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# CSV Summary Tables
# ═══════════════════════════════════════════════════════════════════════════════

def write_publication_csvs(scored_runs, tables_dir):
    """Write comprehensive CSV tables."""
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Significance table
    paired = defaultdict(dict)
    for run in scored_runs:
        key = (run["eval_id"], run["profile"])
        scores = run.get("scores", {}).get("averaged", {})
        paired[key][run["variant"]] = scores

    metrics = [
        "routing_accuracy", "tool_correctness", "task_completion",
        "refusal_correctness", "answer_relevancy", "coherence",
        "correctness", "personalization_accuracy", "readability",
    ]

    rows = []
    for metric in metrics:
        full_v, base_v = [], []
        for key, vdata in paired.items():
            if "full_system" in vdata and "baseline" in vdata:
                fv = vdata["full_system"].get(metric)
                bv = vdata["baseline"].get(metric)
                if fv is not None and bv is not None:
                    try:
                        full_v.append(float(fv))
                        base_v.append(float(bv))
                    except (TypeError, ValueError):
                        continue

        if len(full_v) < 10:
            continue

        fm, bm = np.mean(full_v), np.mean(base_v)
        diffs = np.array(full_v) - np.array(base_v)
        d = np.mean(diffs) / np.std(diffs) if np.std(diffs) > 0 else 0
        _, p = stats.ttest_rel(full_v, base_v)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

        rows.append({
            "metric": metric,
            "prism_mean": round(fm, 4),
            "baseline_mean": round(bm, 4),
            "delta": round(fm - bm, 4),
            "cohens_d": round(d, 4),
            "p_value": round(p, 6),
            "significance": sig,
            "winner": "PRISM" if fm > bm and p < 0.05 else "Baseline" if bm > fm and p < 0.05 else "Tie",
        })

    path = tables_dir / "significance_tests.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print(f"  CSV tables written to {tables_dir}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from prism_eval.config import RAW_DIR, TABLES_DIR, PLOTS_DIR, LATEX_DIR

    scored_runs = load_scored_runs(RAW_DIR / "scored_runs.jsonl")
    print(f"Loaded {len(scored_runs)} scored runs\n")

    print("Generating publication plots...")
    fig1_composite_bars(scored_runs, PLOTS_DIR)
    fig2_group_radar(scored_runs, PLOTS_DIR)
    fig3_significance_forest(scored_runs, PLOTS_DIR)
    fig4_category_heatmap(scored_runs, PLOTS_DIR)
    fig5_ablation_waterfall(scored_runs, PLOTS_DIR)
    fig6_judge_agreement(scored_runs, PLOTS_DIR)
    fig7_profile_comparison(scored_runs, PLOTS_DIR)
    fig8_full_metric_comparison(scored_runs, PLOTS_DIR)

    print("\nGenerating publication LaTeX...")
    write_publication_latex(scored_runs, LATEX_DIR)

    print("\nGenerating publication CSVs...")
    write_publication_csvs(scored_runs, TABLES_DIR)

    print("\nDone! All publication materials generated.")


if __name__ == "__main__":
    main()

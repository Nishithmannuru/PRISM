"""Generate evaluation plots from aggregated scores."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


# Color palette for variants
VARIANT_COLORS = {
    "full_system": "#2ecc71",
    "baseline": "#e74c3c",
    "no_rag": "#e67e22",
    "no_personalization": "#3498db",
    "no_internal_eval": "#9b59b6",
    "no_web_search": "#f39c12",
    "no_query_refinement": "#1abc9c",
}


def write_plots(aggregated: dict, plots_dir: Path):
    """Generate all evaluation plots."""
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_variant_comparison(aggregated, plots_dir)
    _plot_radar_chart(aggregated, plots_dir)
    _plot_category_breakdown(aggregated, plots_dir)
    _plot_delta_heatmap(aggregated, plots_dir)
    _plot_profile_comparison(aggregated, plots_dir)

    print(f"  Plots written to {plots_dir}")


def _plot_variant_comparison(agg: dict, out_dir: Path):
    """Bar chart comparing all variants across key metrics."""
    per_variant = agg.get("per_variant", {})
    if not per_variant:
        return

    key_metrics = [
        "faithfulness", "answer_relevancy", "correctness",
        "routing_accuracy", "tool_correctness", "task_completion",
        "coherence", "readability",
    ]
    available = [m for m in key_metrics if any(m in v for v in per_variant.values())]
    if not available:
        return

    variants = sorted(per_variant.keys())
    x = np.arange(len(available))
    width = 0.8 / len(variants)

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, variant in enumerate(variants):
        means = []
        stds = []
        for m in available:
            stats = per_variant[variant].get(m, {})
            means.append(stats.get("mean", 0))
            stds.append(stats.get("std", 0))

        color = VARIANT_COLORS.get(variant, "#95a5a6")
        ax.bar(x + i * width, means, width, yerr=stds,
               label=variant.replace("_", " "), color=color, alpha=0.85,
               capsize=2)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("PRISM Evaluation: Variant Comparison")
    ax.set_xticks(x + width * len(variants) / 2)
    ax.set_xticklabels([m.replace("_", "\n") for m in available], fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "variant_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "variant_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_radar_chart(agg: dict, out_dir: Path):
    """Radar chart for full_system vs baseline."""
    per_variant = agg.get("per_variant", {})
    full = per_variant.get("full_system", {})
    baseline = per_variant.get("baseline", {})
    if not full or not baseline:
        return

    metrics = [m for m in full.keys() if m in baseline]
    if len(metrics) < 3:
        return

    # Select top metrics for readability
    priority = [
        "faithfulness", "answer_relevancy", "correctness", "hallucination",
        "routing_accuracy", "tool_correctness", "task_completion",
        "coherence", "readability",
    ]
    metrics = [m for m in priority if m in full and m in baseline]
    if len(metrics) < 3:
        return

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    full_vals = [full[m]["mean"] for m in metrics] + [full[metrics[0]]["mean"]]
    base_vals = [baseline[m]["mean"] for m in metrics] + [baseline[metrics[0]]["mean"]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, full_vals, alpha=0.25, color="#2ecc71")
    ax.plot(angles, full_vals, "o-", color="#2ecc71", linewidth=2, label="Full System")
    ax.fill(angles, base_vals, alpha=0.25, color="#e74c3c")
    ax.plot(angles, base_vals, "o-", color="#e74c3c", linewidth=2, label="Baseline (LLM only)")

    ax.set_thetagrids(
        [a * 180 / np.pi for a in angles[:-1]],
        [m.replace("_", "\n") for m in metrics],
        fontsize=9,
    )
    ax.set_ylim(0, 1)
    ax.set_title("PRISM Full System vs Baseline", pad=20, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    fig.savefig(out_dir / "radar_full_vs_baseline.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "radar_full_vs_baseline.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_category_breakdown(agg: dict, out_dir: Path):
    """Per-category performance for full_system."""
    per_vc = agg.get("per_variant_category", {})
    if not per_vc:
        return

    categories = sorted(set(k.split("|")[1] for k in per_vc.keys() if "full_system" in k))
    key_metrics = ["faithfulness", "correctness", "routing_accuracy", "task_completion"]
    available = []
    for m in key_metrics:
        for cat in categories:
            if m in per_vc.get(f"full_system|{cat}", {}):
                available.append(m)
                break

    if not available or not categories:
        return

    x = np.arange(len(categories))
    width = 0.8 / len(available)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#2ecc71", "#3498db", "#e67e22", "#9b59b6"]

    for i, metric in enumerate(available):
        means = []
        stds = []
        for cat in categories:
            stats = per_vc.get(f"full_system|{cat}", {}).get(metric, {})
            means.append(stats.get("mean", 0))
            stds.append(stats.get("std", 0))

        ax.bar(x + i * width, means, width, yerr=stds,
               label=metric.replace("_", " "), color=colors[i % len(colors)],
               alpha=0.85, capsize=2)

    ax.set_xlabel("Question Category")
    ax.set_ylabel("Score")
    ax.set_title("Full System Performance by Category")
    ax.set_xticks(x + width * len(available) / 2)
    ax.set_xticklabels([c.replace("_", " ") for c in categories])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "category_breakdown.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "category_breakdown.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_delta_heatmap(agg: dict, out_dir: Path):
    """Heatmap showing performance delta of each variant vs full_system."""
    deltas = agg.get("deltas", {})
    if not deltas:
        return

    variants = sorted(deltas.keys())
    all_metrics = sorted(set(m for v in deltas.values() for m in v.keys()))

    if not variants or not all_metrics:
        return

    # Select key metrics
    priority = [
        "faithfulness", "answer_relevancy", "correctness", "hallucination",
        "routing_accuracy", "tool_correctness", "task_completion",
        "coherence", "readability",
    ]
    metrics = [m for m in priority if m in all_metrics]
    if not metrics:
        metrics = all_metrics[:10]

    data = np.zeros((len(variants), len(metrics)))
    for i, v in enumerate(variants):
        for j, m in enumerate(metrics):
            data[i, j] = deltas[v].get(m, 0)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.5)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(variants)))
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=8)
    ax.set_yticklabels([v.replace("_", " ") for v in variants], fontsize=9)

    # Add value annotations
    for i in range(len(variants)):
        for j in range(len(metrics)):
            val = data[i, j]
            color = "white" if abs(val) > 0.3 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    color=color, fontsize=8)

    ax.set_title("Performance Delta vs Full System (green = full system better)")
    fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    fig.savefig(out_dir / "delta_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "delta_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_profile_comparison(agg: dict, out_dir: Path):
    """Compare metrics across student profiles (undergrad, masters, phd)."""
    per_profile = agg.get("per_profile", {})
    if not per_profile:
        return

    profiles = sorted(per_profile.keys())
    key_metrics = ["readability", "personalization_accuracy", "coherence", "correctness"]
    available = [m for m in key_metrics if any(m in per_profile[p] for p in profiles)]

    if not available:
        return

    x = np.arange(len(available))
    width = 0.25
    colors = {"undergrad": "#3498db", "masters": "#2ecc71", "phd": "#e74c3c"}

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, profile in enumerate(profiles):
        means = [per_profile[profile].get(m, {}).get("mean", 0) for m in available]
        stds = [per_profile[profile].get(m, {}).get("std", 0) for m in available]
        ax.bar(x + i * width, means, width, yerr=stds,
               label=profile, color=colors.get(profile, "#95a5a6"),
               alpha=0.85, capsize=2)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Performance by Student Profile")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.replace("_", " ") for m in available])
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_dir / "profile_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "profile_comparison.pdf", bbox_inches="tight")
    plt.close(fig)

"""Matplotlib bar charts: overall metric comparison, per-category top metrics. PNG + PDF."""

import logging
from pathlib import Path
from typing import Dict, Any, List

from eval_runner.config import PLOTS_DIR

logger = logging.getLogger(__name__)


def _ensure_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plots")
        return None


def write_plots(aggregates: Dict[str, Any], results_dir: Path) -> None:
    plt = _ensure_matplotlib()
    if plt is None:
        return
    plots_dir = results_dir / PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    per_variant = aggregates.get("per_variant", {})
    deltas = aggregates.get("deltas", {})
    metrics_list = []
    for v in per_variant:
        for m in per_variant[v]:
            if m not in metrics_list:
                metrics_list.append(m)
    metrics_list = sorted(metrics_list)

    # One plot per metric: bar chart (variants x mean, error bars = std)
    for metric in metrics_list:
        variants = []
        means = []
        stds = []
        for v in sorted(per_variant.keys()):
            s = per_variant[v].get(metric, {})
            if s.get("n", 0) == 0:
                continue
            variants.append(v)
            means.append(s["mean"])
            stds.append(s.get("std", 0))
        if not variants:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        x = range(len(variants))
        ax.bar(x, means, yerr=stds, capsize=4)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=45, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(f"Metric: {metric}")
        ax.set_ylim(0, 1.1)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = plots_dir / f"overall_{metric}.{ext}"
            fig.savefig(path, bbox_inches="tight")
            logger.info(f"Wrote {path}")
        plt.close(fig)

    # Per-category bar chart for top metrics
    top_metrics = ["correctness", "groundedness", "context_recall", "context_precision", "tool_correctness"]
    per_variant_category = aggregates.get("per_variant_category", {})
    import numpy as np
    for category in sorted(set(k[1] for k in per_variant_category)):
        # Build (variant, metric) means for this category
        metrics_present = [m for m in top_metrics if any(per_variant_category.get((v, category), {}).get(m, {}).get("n") for v in per_variant)]
        if not metrics_present:
            continue
        fig, ax = plt.subplots(figsize=(10, 5))
        variants = sorted(per_variant.keys())
        n_metrics = len(metrics_present)
        x = np.arange(n_metrics)
        width = 0.8 / max(len(variants), 1)
        for i, variant in enumerate(variants):
            vals = [per_variant_category.get((variant, category), {}).get(m, {}).get("mean", 0) for m in metrics_present]
            offset = (i - len(variants) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=variant)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_present, rotation=45, ha="right")
        ax.set_ylabel("Score")
        ax.set_title(f"Category: {category}")
        ax.legend()
        ax.set_ylim(0, 1.1)
        fig.tight_layout()
        safe_cat = category.replace(" ", "_").replace("/", "_")[:30]
        for ext in ("png", "pdf"):
            path = plots_dir / f"category_{safe_cat}.{ext}"
            fig.savefig(path, bbox_inches="tight")
            logger.info(f"Wrote {path}")
        plt.close(fig)

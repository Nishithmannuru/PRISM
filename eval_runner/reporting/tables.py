"""Write CSV summary tables: per variant, per category, deltas."""

import csv
import logging
from pathlib import Path
from typing import Dict, Any, List

from eval_runner.config import TABLES_DIR, LATEX_DIR

logger = logging.getLogger(__name__)


def write_tables(aggregates: Dict[str, Any], results_dir: Path) -> None:
    tables_dir = results_dir / TABLES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Per-variant overall
    per_variant = aggregates.get("per_variant", {})
    path = tables_dir / "per_variant_overall.csv"
    rows = []
    metrics_seen = set()
    for v in per_variant:
        for m in per_variant[v]:
            metrics_seen.add(m)
    metrics_list = sorted(metrics_seen)
    rows.append(["variant"] + [f"{m}_mean" for m in metrics_list] + [f"{m}_std" for m in metrics_list])
    for variant in sorted(per_variant.keys()):
        row = [variant]
        for m in metrics_list:
            s = per_variant[variant].get(m, {})
            row.append(f"{s.get('mean', 0):.4f}")
        for m in metrics_list:
            s = per_variant[variant].get(m, {})
            row.append(f"{s.get('std', 0):.4f}")
        rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    logger.info(f"Wrote {path}")

    # Optional LaTeX table (per-variant overall)
    latex_dir = results_dir / LATEX_DIR
    latex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = latex_dir / "per_variant_overall.tex"
    with open(tex_path, "w") as f:
        f.write("\\begin{tabular}{l" + "cc" * len(metrics_list) + "}\n\\hline\n")
        f.write("Variant & " + " & ".join([f"{m} (mean $\\pm$ std)" for m in metrics_list]) + " \\\\\n\\hline\n")
        for variant in sorted(per_variant.keys()):
            row = [variant]
            for m in metrics_list:
                s = per_variant[variant].get(m, {})
                row.append(f"{s.get('mean', 0):.3f} $\\pm$ {s.get('std', 0):.3f}")
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    logger.info(f"Wrote {tex_path}")

    # Per-category
    per_category = aggregates.get("per_category", {})
    path = tables_dir / "per_category.csv"
    metrics_seen = set()
    for c in per_category:
        for m in per_category[c]:
            metrics_seen.add(m)
    metrics_list = sorted(metrics_seen)
    rows = [["category"] + [f"{m}_mean" for m in metrics_list] + [f"{m}_std" for m in metrics_list]]
    for category in sorted(per_category.keys()):
        row = [category]
        for m in metrics_list:
            s = per_category[category].get(m, {})
            row.append(f"{s.get('mean', 0):.4f}")
        for m in metrics_list:
            s = per_category[category].get(m, {})
            row.append(f"{s.get('std', 0):.4f}")
        rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    logger.info(f"Wrote {path}")

    # Deltas (full_system - variant)
    deltas = aggregates.get("deltas", {})
    path = tables_dir / "deltas_full_system_minus_variant.csv"
    metrics_seen = set()
    for v in deltas:
        for m in deltas[v]:
            metrics_seen.add(m)
    metrics_list = sorted(metrics_seen)
    rows = [["variant"] + metrics_list]
    for variant in sorted(deltas.keys()):
        row = [variant] + [f"{deltas[variant].get(m, 0):.4f}" for m in metrics_list]
        rows.append(row)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    logger.info(f"Wrote {path}")

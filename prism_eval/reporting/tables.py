"""Generate CSV and LaTeX tables from aggregated scores."""

import csv
from pathlib import Path


def write_tables(aggregated: dict, tables_dir: Path, latex_dir: Path):
    """Write all tables to CSV and LaTeX."""
    _write_per_variant_csv(aggregated, tables_dir)
    _write_per_category_csv(aggregated, tables_dir)
    _write_per_course_csv(aggregated, tables_dir)
    _write_per_profile_csv(aggregated, tables_dir)
    _write_deltas_csv(aggregated, tables_dir)
    _write_per_variant_latex(aggregated, latex_dir)
    _write_deltas_latex(aggregated, latex_dir)
    print(f"  Tables written to {tables_dir}")


def _write_per_variant_csv(agg: dict, out_dir: Path):
    per_variant = agg.get("per_variant", {})
    if not per_variant:
        return

    all_metrics = sorted(
        set(m for v in per_variant.values() for m in v.keys())
    )
    path = out_dir / "per_variant_overall.csv"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant"] + all_metrics)
        for variant in sorted(per_variant.keys()):
            row = [variant]
            for m in all_metrics:
                stats = per_variant[variant].get(m, {})
                mean = stats.get("mean", "")
                std = stats.get("std", "")
                if mean != "":
                    row.append(f"{mean:.4f} ± {std:.4f}")
                else:
                    row.append("")
            w.writerow(row)


def _write_per_category_csv(agg: dict, out_dir: Path):
    per_cat = agg.get("per_category", {})
    if not per_cat:
        return

    all_metrics = sorted(set(m for v in per_cat.values() for m in v.keys()))
    path = out_dir / "per_category.csv"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category"] + all_metrics)
        for cat in sorted(per_cat.keys()):
            row = [cat]
            for m in all_metrics:
                stats = per_cat[cat].get(m, {})
                mean = stats.get("mean", "")
                std = stats.get("std", "")
                if mean != "":
                    row.append(f"{mean:.4f} ± {std:.4f}")
                else:
                    row.append("")
            w.writerow(row)


def _write_per_course_csv(agg: dict, out_dir: Path):
    per_course = agg.get("per_course", {})
    if not per_course:
        return

    all_metrics = sorted(set(m for v in per_course.values() for m in v.keys()))
    path = out_dir / "per_course.csv"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["course"] + all_metrics)
        for course in sorted(per_course.keys()):
            row = [course]
            for m in all_metrics:
                stats = per_course[course].get(m, {})
                mean = stats.get("mean", "")
                std = stats.get("std", "")
                if mean != "":
                    row.append(f"{mean:.4f} ± {std:.4f}")
                else:
                    row.append("")
            w.writerow(row)


def _write_per_profile_csv(agg: dict, out_dir: Path):
    per_profile = agg.get("per_profile", {})
    if not per_profile:
        return

    all_metrics = sorted(set(m for v in per_profile.values() for m in v.keys()))
    path = out_dir / "per_profile.csv"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["profile"] + all_metrics)
        for profile in sorted(per_profile.keys()):
            row = [profile]
            for m in all_metrics:
                stats = per_profile[profile].get(m, {})
                mean = stats.get("mean", "")
                std = stats.get("std", "")
                if mean != "":
                    row.append(f"{mean:.4f} ± {std:.4f}")
                else:
                    row.append("")
            w.writerow(row)


def _write_deltas_csv(agg: dict, out_dir: Path):
    deltas = agg.get("deltas", {})
    if not deltas:
        return

    all_metrics = sorted(set(m for v in deltas.values() for m in v.keys()))
    path = out_dir / "deltas_vs_full_system.csv"

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["variant"] + all_metrics)
        for variant in sorted(deltas.keys()):
            row = [variant]
            for m in all_metrics:
                delta = deltas[variant].get(m, "")
                if delta != "":
                    row.append(f"{delta:+.4f}")
                else:
                    row.append("")
            w.writerow(row)


def _write_per_variant_latex(agg: dict, out_dir: Path):
    per_variant = agg.get("per_variant", {})
    if not per_variant:
        return

    # Select key metrics for LaTeX table
    key_metrics = [
        "faithfulness", "answer_relevancy", "correctness", "hallucination",
        "routing_accuracy", "tool_correctness", "task_completion",
        "coherence", "readability", "toxicity", "bias",
    ]
    available = [m for m in key_metrics if any(m in v for v in per_variant.values())]

    path = out_dir / "per_variant_overall.tex"
    with open(path, "w") as f:
        cols = "l" + "c" * len(available)
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Per-Variant Evaluation Results (Mean ± Std)}}\n")
        f.write(f"\\label{{tab:per_variant}}\n")
        f.write(f"\\begin{{tabular}}{{{cols}}}\n")
        f.write("\\toprule\n")

        # Header
        header = "Variant & " + " & ".join(
            m.replace("_", " ").title()[:12] for m in available
        )
        f.write(header + " \\\\\n")
        f.write("\\midrule\n")

        for variant in sorted(per_variant.keys()):
            name = variant.replace("_", "\\_")
            cells = [name]
            for m in available:
                stats = per_variant[variant].get(m, {})
                mean = stats.get("mean")
                std = stats.get("std")
                if mean is not None:
                    cells.append(f"{mean:.3f} ± {std:.3f}")
                else:
                    cells.append("—")
            f.write(" & ".join(cells) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def _write_deltas_latex(agg: dict, out_dir: Path):
    deltas = agg.get("deltas", {})
    if not deltas:
        return

    key_metrics = [
        "faithfulness", "answer_relevancy", "correctness", "hallucination",
        "routing_accuracy", "tool_correctness", "coherence", "readability",
    ]
    available = [m for m in key_metrics if any(m in v for v in deltas.values())]

    path = out_dir / "deltas_vs_full_system.tex"
    with open(path, "w") as f:
        cols = "l" + "c" * len(available)
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{Performance Delta vs Full System (positive = full system better)}}\n")
        f.write(f"\\label{{tab:deltas}}\n")
        f.write(f"\\begin{{tabular}}{{{cols}}}\n")
        f.write("\\toprule\n")

        header = "Variant & " + " & ".join(
            m.replace("_", " ").title()[:12] for m in available
        )
        f.write(header + " \\\\\n")
        f.write("\\midrule\n")

        for variant in sorted(deltas.keys()):
            name = variant.replace("_", "\\_")
            cells = [name]
            for m in available:
                delta = deltas[variant].get(m)
                if delta is not None:
                    sign = "+" if delta > 0 else ""
                    cells.append(f"{sign}{delta:.3f}")
                else:
                    cells.append("—")
            f.write(" & ".join(cells) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

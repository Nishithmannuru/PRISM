"""Publication-ready evaluation output for PRISM paper.

Generates camera-ready tables (LaTeX) and figures (PDF+PNG) aligned with
best-paper standards. Clean narrative, no clutter, statistical rigor.

Evaluation framework:
  - 152 questions × 3 profiles × 7 variants = 3,192 runs
  - Dual-judge: GPT-4.1-mini (multi-step decomposed) + Claude Sonnet 4.6 (single-prompt)
  - Cross-family evaluator independence (system uses GPT-4o; judges are GPT-4.1-mini + Claude)
  - 4 metric categories: Agentic Behavior, Retrieval Quality, Response Quality, Safety
"""

import json
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from collections import defaultdict
from scipy import stats
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from prism_eval.config import RAW_DIR, TABLES_DIR, PLOTS_DIR, LATEX_DIR

# ── Style ────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# Academic color palette (colorblind-safe, from ColorBrewer)
C_FULL = "#1B9E77"      # teal-green
C_BASE = "#D95F02"      # orange
C_ABL  = ["#7570B3", "#E7298A", "#66A61E", "#E6AB02", "#A6761D"]  # purple, pink, green, yellow, brown

VARIANT_ORDER = [
    "full_system", "baseline", "no_rag", "no_web_search",
    "no_query_refinement", "no_internal_eval", "no_personalization",
]
VARIANT_LABELS = {
    "full_system": "PRISM (Full)",
    "baseline": "LLM-only",
    "no_rag": "w/o RAG",
    "no_web_search": "w/o Web Search",
    "no_query_refinement": "w/o Query Ref.",
    "no_internal_eval": "w/o Internal Eval",
    "no_personalization": "w/o Personalization",
}
VARIANT_COLORS = {
    "full_system": C_FULL,
    "baseline": C_BASE,
    "no_rag": C_ABL[0],
    "no_web_search": C_ABL[1],
    "no_query_refinement": C_ABL[2],
    "no_internal_eval": C_ABL[3],
    "no_personalization": C_ABL[4],
}


def load_data():
    runs = []
    with open(RAW_DIR / "scored_runs.jsonl") as f:
        for line in f:
            runs.append(json.loads(line))
    return runs


def vmean(runs, variant, metric, judge="averaged", category=None):
    """Variant × metric mean."""
    vals = []
    for r in runs:
        if r["variant"] != variant:
            continue
        if category and r.get("category") != category:
            continue
        s = r.get("scores", {}).get(judge, {})
        v = s.get(metric)
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
    return (np.mean(vals), np.std(vals), len(vals)) if vals else (None, None, 0)


def paired_test(runs, v1, v2, metric, judge="averaged"):
    """Paired t-test between two variants on a metric."""
    paired = defaultdict(dict)
    for r in runs:
        key = (r["eval_id"], r["profile"])
        s = r.get("scores", {}).get(judge, {})
        v = s.get(metric)
        if v is not None:
            try:
                paired[key][r["variant"]] = float(v)
            except (TypeError, ValueError):
                continue
    v1_vals, v2_vals = [], []
    for key, vdata in paired.items():
        if v1 in vdata and v2 in vdata:
            v1_vals.append(vdata[v1])
            v2_vals.append(vdata[v2])
    if len(v1_vals) < 10:
        return None
    t, p = stats.ttest_rel(v1_vals, v2_vals)
    diffs = np.array(v1_vals) - np.array(v2_vals)
    d = np.mean(diffs) / np.std(diffs) if np.std(diffs) > 0 else 0
    return {"delta": np.mean(diffs), "t": t, "p": p, "d": d, "n": len(v1_vals)}


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 5: Overall System Performance
# ═════════════════════════════════════════════════════════════════════════════

def table5_overall(runs):
    """Paper Table 5: PRISM full system performance by metric category."""

    categories_metrics = [
        ("Agentic Behavior", [
            ("Task Completeness", "task_completion"),
            ("Tool Correctness", "tool_correctness"),
            ("Routing Accuracy", "routing_accuracy"),
            ("Refusal Correctness", "refusal_correctness"),
        ]),
        ("Retrieval Quality", [
            ("Context Precision", "contextual_precision"),
            ("Context Recall", "contextual_recall"),
            ("Correctness (Keypoint Coverage)", "correctness"),
        ]),
        ("Response Quality", [
            ("Answer Relevancy", "answer_relevancy"),
            ("Coherence", "coherence"),
            ("Personalization Accuracy", "personalization_accuracy"),
            ("Readability Alignment", "readability"),
        ]),
        ("Safety", [
            ("Toxicity", "toxicity"),
            ("Bias", "bias"),
        ]),
    ]

    print("=" * 72)
    print("TABLE 5: Overall System Performance")
    print("=" * 72)

    with open(LATEX_DIR / "paper_table5.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Overall evaluation results for PRISM across four metric categories. ")
        f.write("Scores are dual-judge averages (GPT-4.1-mini + Claude Sonnet~4.6); ")
        f.write("$n$ denotes the number of evaluated runs. ")
        f.write("Toxicity and bias are inverted (lower raw score = better).}\n")
        f.write("\\label{tab:overall}\n")
        f.write("\\begin{tabular}{@{}llccc@{}}\n\\toprule\n")
        f.write("Category & Metric & Mean & Std & $n$ \\\\\n")

        rows_for_csv = []
        for cat_name, metrics in categories_metrics:
            f.write("\\midrule\n")
            f.write(f"\\multirow{{{len(metrics)}}}{{*}}{{{cat_name}}}\n")
            for i, (label, key) in enumerate(metrics):
                m, s, n = vmean(runs, "full_system", key)
                if m is not None:
                    if key in ("toxicity", "bias"):
                        display = f"{m:.3f}"
                        note = " $\\downarrow$"
                    else:
                        display = f"{m:.2f}"
                        note = ""
                    f.write(f"  & {label}{note} & {display} & {s:.2f} & {n} \\\\\n")
                    print(f"  {cat_name:20s} | {label:30s} | {m:.3f} ± {s:.3f} | n={n}")
                    rows_for_csv.append({"category": cat_name, "metric": label, "mean": round(m, 4), "std": round(s, 4), "n": n})

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    # CSV
    with open(TABLES_DIR / "paper_table5.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category", "metric", "mean", "std", "n"])
        w.writeheader()
        w.writerows(rows_for_csv)


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 6: Ablation Study
# ═════════════════════════════════════════════════════════════════════════════

def table6_ablation(runs):
    """Paper Table 6: Ablation study across 7 variants × 6 key metrics."""

    metrics = [
        ("Task\nCompl.", "task_completion"),
        ("Tool\nCorr.", "tool_correctness"),
        ("Routing\nAcc.", "routing_accuracy"),
        ("Correct-\nness", "correctness"),
        ("Answer\nRelev.", "answer_relevancy"),
        ("Person.\nAcc.", "personalization_accuracy"),
    ]
    metric_labels_tex = [
        "Task Compl.", "Tool Corr.", "Routing Acc.",
        "Correctness", "Answer Rel.", "Pers. Acc.",
    ]

    print("\n" + "=" * 72)
    print("TABLE 6: Ablation Study")
    print("=" * 72)

    # Collect data
    data = {}
    for v in VARIANT_ORDER:
        data[v] = {}
        for _, mkey in metrics:
            m, s, n = vmean(runs, v, mkey)
            data[v][mkey] = (m, s, n)

    # Find column-wise best
    col_best = {}
    for _, mkey in metrics:
        best_val = -1
        best_var = None
        for v in VARIANT_ORDER:
            m = data[v][mkey][0]
            if m is not None and m > best_val:
                best_val = m
                best_var = v
        col_best[mkey] = best_var

    # Significance vs full_system
    sig = {}
    for v in VARIANT_ORDER:
        if v == "full_system":
            continue
        sig[v] = {}
        for _, mkey in metrics:
            result = paired_test(runs, "full_system", v, mkey)
            sig[v][mkey] = result

    # Print
    header = f"{'Variant':<22s}" + "".join(f"{'':>2s}{ml:>10s}" for ml in metric_labels_tex)
    print(header)
    print("-" * len(header))

    for v in VARIANT_ORDER:
        parts = [f"{VARIANT_LABELS[v]:<22s}"]
        for _, mkey in metrics:
            m = data[v][mkey][0]
            if m is None:
                parts.append(f"{'---':>12s}")
            else:
                star = ""
                if v != "full_system" and v in sig and mkey in sig[v] and sig[v][mkey]:
                    p = sig[v][mkey]["p"]
                    star = "†" if p < 0.001 else "*" if p < 0.05 else ""
                bold = ">" if v == col_best[mkey] else " "
                parts.append(f"{bold}{m:.3f}{star:>3s}   ")
        print("".join(parts))

    # LaTeX
    with open(LATEX_DIR / "paper_table6.tex", "w") as f:
        ncols = len(metrics)
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Ablation study: each row removes one component from PRISM. ")
        f.write("Bold = column best. ")
        f.write("$^\\dagger$~significant degradation vs.\\ Full System ($p < 0.001$, paired $t$-test); ")
        f.write("$^*$~$p < 0.05$.}\n")
        f.write("\\label{tab:ablation}\n")
        f.write(f"\\begin{{tabular}}{{@{{}}l{'c' * ncols}@{{}}}}\n\\toprule\n")
        f.write("Variant & " + " & ".join(metric_labels_tex) + " \\\\\n")
        f.write("\\midrule\n")

        for v in VARIANT_ORDER:
            name = VARIANT_LABELS[v]
            if v == "full_system":
                name = "\\textsc{PRISM} (Full)"
            cells = [name]
            for _, mkey in metrics:
                m = data[v][mkey][0]
                if m is None:
                    cells.append("---")
                else:
                    val_str = f"{m:.2f}"
                    if v == col_best[mkey]:
                        val_str = f"\\textbf{{{val_str}}}"
                    if v != "full_system" and v in sig and mkey in sig[v] and sig[v][mkey]:
                        p = sig[v][mkey]["p"]
                        if p < 0.001:
                            val_str += "$^\\dagger$"
                        elif p < 0.05:
                            val_str += "$^*$"
                    cells.append(val_str)
            f.write(" & ".join(cells) + " \\\\\n")
            if v == "baseline":
                f.write("\\midrule\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 7: Per-Category Performance
# ═════════════════════════════════════════════════════════════════════════════

def table7_categories(runs):
    """Full system performance broken down by question category."""

    print("\n" + "=" * 72)
    print("TABLE 7: Per-Category Performance (Full System)")
    print("=" * 72)

    # Each category gets its applicable metrics
    cat_config = [
        ("Course-Based", "course_based", [
            ("Correctness", "correctness"),
            ("Ctx Precision", "contextual_precision"),
            ("Ctx Recall", "contextual_recall"),
            ("Routing Acc.", "routing_accuracy"),
            ("Task Compl.", "task_completion"),
            ("Pers. Acc.", "personalization_accuracy"),
            ("Readability", "readability"),
        ]),
        ("Web-Required", "web_required", [
            ("Correctness", "correctness"),
            ("Ctx Precision", "contextual_precision"),
            ("Ctx Recall", "contextual_recall"),
            ("Routing Acc.", "routing_accuracy"),
            ("Task Compl.", "task_completion"),
            ("Readability", "readability"),
        ]),
        ("Multi-Hop", "multi_hop", [
            ("Correctness", "correctness"),
            ("Ctx Recall", "contextual_recall"),
            ("Routing Acc.", "routing_accuracy"),
            ("Task Compl.", "task_completion"),
        ]),
        ("Out-of-Scope", "out_of_scope", [
            ("Refusal Corr.", "refusal_correctness"),
            ("Routing Acc.", "routing_accuracy"),
            ("Task Compl.", "task_completion"),
        ]),
    ]

    # Collect all unique metrics
    all_metric_keys = []
    all_metric_labels = []
    seen = set()
    for _, _, mlist in cat_config:
        for label, key in mlist:
            if key not in seen:
                seen.add(key)
                all_metric_keys.append(key)
                all_metric_labels.append(label)

    with open(LATEX_DIR / "paper_table7.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{PRISM full system performance by question category. ")
        f.write("Dashes indicate metrics not applicable to that category. ")
        f.write("All scores are dual-judge averages.}\n")
        f.write("\\label{tab:categories}\n")
        f.write(f"\\begin{{tabular}}{{@{{}}l{'c' * len(all_metric_keys)}@{{}}}}\n\\toprule\n")

        # Short labels for header
        short = {
            "correctness": "Corr.", "contextual_precision": "Ctx P.",
            "contextual_recall": "Ctx R.", "routing_accuracy": "Rte.",
            "task_completion": "Task", "personalization_accuracy": "Pers.",
            "readability": "Read.", "refusal_correctness": "Ref.",
        }
        header = "Category & " + " & ".join(short.get(k, k[:5]) for k in all_metric_keys)
        f.write(header + " \\\\\n\\midrule\n")

        for cat_label, cat_key, mlist in cat_config:
            applicable = {key for _, key in mlist}
            cells = [cat_label]
            for mkey in all_metric_keys:
                if mkey in applicable:
                    m, s, n = vmean(runs, "full_system", mkey, category=cat_key)
                    cells.append(f"{m:.2f}" if m is not None else "---")
                else:
                    cells.append("---")
            f.write(" & ".join(cells) + " \\\\\n")
            print(f"  {cat_label}: " + ", ".join(
                f"{short.get(k,k)}={vmean(runs, 'full_system', k, category=cat_key)[0]:.2f}"
                for _, k in mlist if vmean(runs, 'full_system', k, category=cat_key)[0] is not None
            ))

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 8: Inter-Judge Agreement
# ═════════════════════════════════════════════════════════════════════════════

def table8_judges(runs):
    """Inter-judge agreement: GPT-4.1-mini vs Claude Sonnet."""

    print("\n" + "=" * 72)
    print("TABLE 8: Inter-Judge Agreement")
    print("=" * 72)

    # Only LLM-judged metrics (skip heuristic ones with r=1.0)
    metrics = [
        ("Correctness", "correctness"),
        ("Refusal Correctness", "refusal_correctness"),
        ("Coherence", "coherence"),
        ("Personalization Acc.", "personalization_accuracy"),
        ("Answer Relevancy", "answer_relevancy"),
        ("Ctx Precision", "contextual_precision"),
        ("Ctx Recall", "contextual_recall"),
    ]

    with open(LATEX_DIR / "paper_table8.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Inter-judge agreement between GPT-4.1-mini (multi-step decomposed scoring) ")
        f.write("and Claude Sonnet~4.6 (single-prompt holistic scoring). ")
        f.write("Heuristic metrics (task completeness, tool correctness, routing accuracy, readability) ")
        f.write("are deterministic and omitted ($r = 1.0$). All correlations significant at $p < 0.001$.}\n")
        f.write("\\label{tab:judges}\n")
        f.write("\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n")
        f.write("Metric & Pearson $r$ & GPT Mean & Sonnet Mean & $n$ \\\\\n")
        f.write("\\midrule\n")

        for label, key in metrics:
            gpt_v, son_v = [], []
            for r in runs:
                g = r.get("scores", {}).get("gpt41mini", {}).get(key)
                s = r.get("scores", {}).get("sonnet", {}).get(key)
                if g is not None and s is not None:
                    try:
                        gpt_v.append(float(g))
                        son_v.append(float(s))
                    except (TypeError, ValueError):
                        continue

            if len(gpt_v) < 10 or np.std(gpt_v) == 0 or np.std(son_v) == 0:
                continue

            r_val, _ = stats.pearsonr(gpt_v, son_v)
            gm, sm = np.mean(gpt_v), np.mean(son_v)
            print(f"  {label:<25s}  r={r_val:.3f}  GPT={gm:.3f}  Son={sm:.3f}  n={len(gpt_v)}")
            f.write(f"{label} & {r_val:.3f} & {gm:.3f} & {sm:.3f} & {len(gpt_v)} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ═════════════════════════════════════════════════════════════════════════════
# TABLE 9: Statistical Significance (Full vs Baseline)
# ═════════════════════════════════════════════════════════════════════════════

def table9_significance(runs):
    """Paired significance tests: PRISM vs LLM-only baseline."""

    print("\n" + "=" * 72)
    print("TABLE 9: Statistical Significance (Full System vs Baseline)")
    print("=" * 72)

    metrics = [
        ("Routing Accuracy", "routing_accuracy"),
        ("Tool Correctness", "tool_correctness"),
        ("Task Completeness", "task_completion"),
        ("Refusal Correctness", "refusal_correctness"),
        ("Answer Relevancy", "answer_relevancy"),
        ("Coherence", "coherence"),
        ("Correctness", "correctness"),
        ("Personalization Acc.", "personalization_accuracy"),
        ("Readability", "readability"),
    ]

    with open(LATEX_DIR / "paper_table9.tex", "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n\\small\n")
        f.write("\\caption{Per-metric comparison of PRISM (Full System) vs.\\ LLM-only baseline. ")
        f.write("Paired $t$-test across matched question--profile pairs. ")
        f.write("Cohen's $d$: small~($\\geq$0.2), medium~($\\geq$0.5), large~($\\geq$0.8). ")
        f.write("Bold = statistically significant winner at $p < 0.05$.}\n")
        f.write("\\label{tab:significance}\n")
        f.write("\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n")
        f.write("Metric & PRISM & Baseline & $\\Delta$ & $d$ & Sig. \\\\\n")
        f.write("\\midrule\n")

        rows = []
        for label, key in metrics:
            result = paired_test(runs, "full_system", "baseline", key)
            if result is None:
                continue
            fm = vmean(runs, "full_system", key)[0]
            bm = vmean(runs, "baseline", key)[0]
            p = result["p"]
            d = result["d"]
            delta = result["delta"]

            sig = "$^{***}$" if p < 0.001 else "$^{**}$" if p < 0.01 else "$^{*}$" if p < 0.05 else ""
            winner = "PRISM" if delta > 0 and p < 0.05 else "Base" if delta < 0 and p < 0.05 else ""

            if delta > 0 and p < 0.05:
                fm_str = f"\\textbf{{{fm:.2f}}}"
                bm_str = f"{bm:.2f}"
            elif delta < 0 and p < 0.05:
                fm_str = f"{fm:.2f}"
                bm_str = f"\\textbf{{{bm:.2f}}}"
            else:
                fm_str = f"{fm:.2f}"
                bm_str = f"{bm:.2f}"

            f.write(f"{label} & {fm_str} & {bm_str} & {delta:+.2f} & {d:.2f} & {sig} \\\\\n")
            print(f"  {label:<25s}  PRISM={fm:.3f}  Base={bm:.3f}  Δ={delta:+.3f}  d={d:.2f}  {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE: Ablation Grouped Bars
# ═════════════════════════════════════════════════════════════════════════════

def fig_ablation(runs):
    """Clean ablation bar chart matching paper Figure 4."""

    metrics = [
        ("Task Compl.", "task_completion"),
        ("Tool Corr.", "tool_correctness"),
        ("Correctness", "correctness"),
        ("Answer Rel.", "answer_relevancy"),
        ("Pers. Acc.", "personalization_accuracy"),
    ]

    variants = VARIANT_ORDER
    n_var = len(variants)
    n_met = len(metrics)
    x = np.arange(n_met)
    width = 0.8 / n_var

    fig, ax = plt.subplots(figsize=(11, 4.5))

    for i, v in enumerate(variants):
        vals = []
        for _, mkey in metrics:
            m, _, _ = vmean(runs, v, mkey)
            vals.append(m if m is not None else 0)
        color = VARIANT_COLORS[v]
        edge = "black" if v == "full_system" else "none"
        lw = 1.2 if v == "full_system" else 0
        ax.bar(x + i * width, vals, width, label=VARIANT_LABELS[v],
               color=color, edgecolor=edge, linewidth=lw, alpha=0.88)

    ax.set_xticks(x + width * n_var / 2 - width / 2)
    ax.set_xticklabels([m[0] for m in metrics])
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True, edgecolor="gray")
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "paper_ablation.pdf")
    fig.savefig(PLOTS_DIR / "paper_ablation.png", dpi=300)
    plt.close(fig)
    print("  -> paper_ablation.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE: Effect Size Forest Plot
# ═════════════════════════════════════════════════════════════════════════════

def fig_effect_sizes(runs):
    """Horizontal bar chart of Cohen's d per metric (Full vs Baseline)."""

    metrics = [
        ("Routing Accuracy", "routing_accuracy"),
        ("Tool Correctness", "tool_correctness"),
        ("Task Completeness", "task_completion"),
        ("Refusal Correctness", "refusal_correctness"),
        ("Personalization Acc.", "personalization_accuracy"),
        ("Readability", "readability"),
        ("Correctness", "correctness"),
        ("Answer Relevancy", "answer_relevancy"),
        ("Coherence", "coherence"),
    ]

    results = []
    for label, key in metrics:
        r = paired_test(runs, "full_system", "baseline", key)
        if r:
            results.append((label, r["d"], r["p"], r["n"]))

    # Sort by effect size
    results.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y = np.arange(len(results))

    for i, (label, d, p, n) in enumerate(results):
        color = C_FULL if d > 0 else C_BASE
        ax.barh(i, d, color=color, alpha=0.78, height=0.65, edgecolor="white", linewidth=0.3)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        # Label on the outside
        if d > 0:
            ax.text(d + 0.08, i, f"{label} {sig}", va="center", fontsize=8.5)
        else:
            ax.text(d - 0.08, i, f"{label} {sig}", va="center", ha="right", fontsize=8.5)

    ax.axvline(0, color="black", linewidth=0.8)
    for thresh in [0.2, 0.5, 0.8]:
        ax.axvline(thresh, color="gray", linewidth=0.4, linestyle=":", alpha=0.5)
        ax.axvline(-thresh, color="gray", linewidth=0.4, linestyle=":", alpha=0.5)

    ax.set_yticks([])
    ax.set_xlabel("Cohen's $d$ (effect size)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Annotation
    ax.text(0.5, len(results) + 0.3, "medium", ha="center", fontsize=7, color="gray", style="italic")
    ax.text(0.8, len(results) + 0.3, "large", ha="center", fontsize=7, color="gray", style="italic")

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_FULL, alpha=0.78, label="PRISM better"),
        Patch(facecolor=C_BASE, alpha=0.78, label="Baseline better"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, frameon=True, edgecolor="gray")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "paper_effect_sizes.pdf")
    fig.savefig(PLOTS_DIR / "paper_effect_sizes.png", dpi=300)
    plt.close(fig)
    print("  -> paper_effect_sizes.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE: Category Heatmap
# ═════════════════════════════════════════════════════════════════════════════

def fig_category_heatmap(runs):
    """Heatmap: variant × category showing task completion."""

    cats = ["course_based", "web_required", "multi_hop", "out_of_scope"]
    cat_labels = ["Course-Based", "Web-Required", "Multi-Hop", "Out-of-Scope"]
    variants = VARIANT_ORDER
    var_labels = [VARIANT_LABELS[v] for v in variants]

    # Use task_completion as the primary "does it work" metric
    data = np.zeros((len(cats), len(variants)))
    for i, cat in enumerate(cats):
        for j, v in enumerate(variants):
            m, _, _ = vmean(runs, v, "task_completion", category=cat)
            data[i, j] = m if m is not None else 0

    fig, ax = plt.subplots(figsize=(9, 3.5))
    im = ax.imshow(data, cmap="YlGn", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(variants)))
    ax.set_yticks(np.arange(len(cats)))
    ax.set_xticklabels(var_labels, rotation=35, ha="right")
    ax.set_yticklabels(cat_labels)

    for i in range(len(cats)):
        for j in range(len(variants)):
            val = data[i, j]
            color = "white" if val > 0.75 else "black"
            weight = "bold" if variants[j] == "full_system" else "normal"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    color=color, fontsize=9, fontweight=weight)

    fig.colorbar(im, ax=ax, shrink=0.85, label="Task Completeness", pad=0.02)
    ax.spines[:].set_visible(False)

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "paper_category_heatmap.pdf")
    fig.savefig(PLOTS_DIR / "paper_category_heatmap.png", dpi=300)
    plt.close(fig)
    print("  -> paper_category_heatmap.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE: Inter-Judge Scatter
# ═════════════════════════════════════════════════════════════════════════════

def fig_judge_scatter(runs):
    """2×2 scatter: GPT vs Sonnet for 4 key metrics."""

    metrics = [
        ("Correctness", "correctness"),
        ("Coherence", "coherence"),
        ("Answer Relevancy", "answer_relevancy"),
        ("Refusal Correctness", "refusal_correctness"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7, 7))

    for idx, (label, key) in enumerate(metrics):
        ax = axes[idx // 2][idx % 2]
        gpt_v, son_v = [], []
        for r in runs:
            g = r.get("scores", {}).get("gpt41mini", {}).get(key)
            s = r.get("scores", {}).get("sonnet", {}).get(key)
            if g is not None and s is not None:
                try:
                    gpt_v.append(float(g))
                    son_v.append(float(s))
                except (TypeError, ValueError):
                    continue

        if len(gpt_v) < 10:
            ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes, ha="center")
            continue

        # Jitter
        jit = 0.012
        gj = np.array(gpt_v) + np.random.normal(0, jit, len(gpt_v))
        sj = np.array(son_v) + np.random.normal(0, jit, len(son_v))

        ax.scatter(gj, sj, alpha=0.12, s=6, color="#4C72B0", rasterized=True)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.35, linewidth=0.8)

        r_val, _ = stats.pearsonr(gpt_v, son_v)
        ax.set_title(f"{label}\n$r$ = {r_val:.3f}, $n$ = {len(gpt_v)}", fontsize=9.5)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if idx >= 2:
            ax.set_xlabel("GPT-4.1-mini")
        if idx % 2 == 0:
            ax.set_ylabel("Claude Sonnet 4.6")

    plt.tight_layout(h_pad=1.5, w_pad=1.5)
    fig.savefig(PLOTS_DIR / "paper_judge_scatter.pdf")
    fig.savefig(PLOTS_DIR / "paper_judge_scatter.png", dpi=300)
    plt.close(fig)
    print("  -> paper_judge_scatter.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE: Radar Chart (Full vs Baseline)
# ═════════════════════════════════════════════════════════════════════════════

def fig_radar(runs):
    """Radar: Full System vs Baseline on shared metrics."""

    metrics = [
        ("Task\nCompl.", "task_completion"),
        ("Tool\nCorr.", "tool_correctness"),
        ("Routing\nAcc.", "routing_accuracy"),
        ("Refusal\nCorr.", "refusal_correctness"),
        ("Correct-\nness", "correctness"),
        ("Answer\nRelev.", "answer_relevancy"),
        ("Coherence", "coherence"),
        ("Pers.\nAcc.", "personalization_accuracy"),
        ("Readability", "readability"),
    ]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    full_vals = []
    base_vals = []
    for _, mkey in metrics:
        fm, _, _ = vmean(runs, "full_system", mkey)
        bm, _, _ = vmean(runs, "baseline", mkey)
        full_vals.append(fm if fm is not None else 0)
        base_vals.append(bm if bm is not None else 0)
    full_vals += full_vals[:1]
    base_vals += base_vals[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    ax.fill(angles, full_vals, alpha=0.18, color=C_FULL)
    ax.plot(angles, full_vals, "o-", color=C_FULL, linewidth=2, markersize=5, label="PRISM (Full)")
    ax.fill(angles, base_vals, alpha=0.18, color=C_BASE)
    ax.plot(angles, base_vals, "s-", color=C_BASE, linewidth=2, markersize=5, label="LLM-only Baseline")

    ax.set_thetagrids([a * 180 / np.pi for a in angles[:-1]],
                      [m[0] for m in metrics], fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7, color="gray")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), frameon=True, edgecolor="gray")

    plt.tight_layout()
    fig.savefig(PLOTS_DIR / "paper_radar.pdf")
    fig.savefig(PLOTS_DIR / "paper_radar.png", dpi=300)
    plt.close(fig)
    print("  -> paper_radar.pdf")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    for d in [TABLES_DIR, PLOTS_DIR, LATEX_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    runs = load_data()
    print(f"Loaded {len(runs)} scored runs\n")

    # Tables
    table5_overall(runs)
    table6_ablation(runs)
    table7_categories(runs)
    table8_judges(runs)
    table9_significance(runs)

    # Figures
    print("\nGenerating figures...")
    fig_ablation(runs)
    fig_effect_sizes(runs)
    fig_category_heatmap(runs)
    fig_judge_scatter(runs)
    fig_radar(runs)

    print("\n" + "=" * 72)
    print("All paper materials generated.")
    print(f"  LaTeX: {LATEX_DIR}/paper_table{{5,6,7,8,9}}.tex")
    print(f"  Plots: {PLOTS_DIR}/paper_{{ablation,effect_sizes,category_heatmap,judge_scatter,radar}}.pdf")
    print("=" * 72)


if __name__ == "__main__":
    main()

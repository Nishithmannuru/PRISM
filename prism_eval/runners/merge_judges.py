"""Merge GPT-4.1-mini multi-step and Claude Sonnet single-prompt scores.

Produces:
  - scored_runs.jsonl with both judges' scores + averaged
  - inter_judge_agreement.json with Pearson r per metric
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from prism_eval.config import RAW_DIR


def main():
    # Load GPT scores
    gpt_scores = {}
    with open(RAW_DIR / "scored_runs_gpt.jsonl") as f:
        for line in f:
            r = json.loads(line)
            key = (r["eval_id"], r["variant"], r["profile"])
            gpt_scores[key] = r

    # Load Sonnet scores
    sonnet_scores = {}
    with open(RAW_DIR / "scored_runs_sonnet.jsonl") as f:
        for line in f:
            r = json.loads(line)
            key = (r["eval_id"], r["variant"], r["profile"])
            sonnet_scores[key] = r

    print(f"GPT runs: {len(gpt_scores)}")
    print(f"Sonnet runs: {len(sonnet_scores)}")

    # Find common keys
    common = set(gpt_scores.keys()) & set(sonnet_scores.keys())
    gpt_only = set(gpt_scores.keys()) - common
    sonnet_only = set(sonnet_scores.keys()) - common
    print(f"Common: {len(common)}, GPT-only: {len(gpt_only)}, Sonnet-only: {len(sonnet_only)}")

    # Collect paired scores for inter-judge agreement
    metric_pairs = {}  # metric -> [(gpt_score, sonnet_score), ...]

    # Merge and write
    merged = []
    for key in sorted(common):
        gpt = gpt_scores[key]
        sonnet = sonnet_scores[key]
        gpt_m = gpt["scores"].get("gpt41mini", {})
        sonnet_m = sonnet["scores"].get("sonnet", {})

        # Average scores where both judges have values
        all_metrics = set(list(gpt_m.keys()) + list(sonnet_m.keys()))
        averaged = {}
        for metric in all_metrics:
            g = gpt_m.get(metric)
            s = sonnet_m.get(metric)

            if g is not None and s is not None:
                try:
                    g_val = float(g)
                    s_val = float(s)
                    averaged[metric] = round((g_val + s_val) / 2, 4)

                    if metric not in metric_pairs:
                        metric_pairs[metric] = []
                    metric_pairs[metric].append((g_val, s_val))
                except (TypeError, ValueError):
                    averaged[metric] = g if g is not None else s
            elif g is not None:
                averaged[metric] = g
            elif s is not None:
                averaged[metric] = s

        entry = {
            "eval_id": gpt["eval_id"],
            "variant": gpt["variant"],
            "profile": gpt["profile"],
            "course_id": gpt.get("course_id", ""),
            "category": gpt.get("category", ""),
            "question": gpt.get("question", ""),
            "scores": {
                "gpt41mini": gpt_m,
                "sonnet": sonnet_m,
                "averaged": averaged,
            },
        }
        merged.append(entry)

    # Write merged scored runs
    out_path = RAW_DIR / "scored_runs.jsonl"
    with open(out_path, "w") as f:
        for entry in merged:
            f.write(json.dumps(entry, default=str) + "\n")
    print(f"\nMerged: {len(merged)} runs -> {out_path}")

    # Compute inter-judge agreement
    print("\n=== Inter-Judge Agreement (Pearson r) ===")
    agreement = {}
    for metric in sorted(metric_pairs.keys()):
        pairs = metric_pairs[metric]
        if len(pairs) < 10:
            print(f"  {metric}: too few pairs ({len(pairs)})")
            continue
        gpt_vals = [p[0] for p in pairs]
        sonnet_vals = [p[1] for p in pairs]

        # Check for zero variance
        if np.std(gpt_vals) == 0 or np.std(sonnet_vals) == 0:
            print(f"  {metric}: zero variance (n={len(pairs)})")
            agreement[metric] = {"r": None, "p": None, "n": len(pairs), "note": "zero variance"}
            continue

        r, p = pearsonr(gpt_vals, sonnet_vals)
        agreement[metric] = {
            "r": round(r, 4),
            "p": round(p, 6),
            "n": len(pairs),
            "gpt_mean": round(np.mean(gpt_vals), 4),
            "sonnet_mean": round(np.mean(sonnet_vals), 4),
            "delta": round(np.mean(gpt_vals) - np.mean(sonnet_vals), 4),
        }
        print(f"  {metric:30s}  r={r:.4f}  p={p:.6f}  n={len(pairs):5d}  "
              f"GPT={np.mean(gpt_vals):.3f}  Sonnet={np.mean(sonnet_vals):.3f}  "
              f"Δ={np.mean(gpt_vals) - np.mean(sonnet_vals):+.3f}")

    # Save agreement
    agree_path = RAW_DIR / "inter_judge_agreement.json"
    with open(agree_path, "w") as f:
        json.dump(agreement, f, indent=2)
    print(f"\nAgreement saved: {agree_path}")


if __name__ == "__main__":
    main()

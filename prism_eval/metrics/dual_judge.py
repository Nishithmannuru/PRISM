"""Dual-judge evaluation: runs metrics with both Claude Sonnet and GPT-4.1-mini.

Reports inter-judge agreement (Pearson r) to validate metric robustness.
"""

import numpy as np
from pathlib import Path
from typing import Optional

from prism_eval.metrics.deepeval_metrics import (
    ClaudeSonnetJudge,
    GPT41MiniJudge,
    PRISMMetrics,
    ScoreCache,
)
from prism_eval.config import CACHE_DIR


class DualJudgeEvaluator:
    """Runs all metrics with two independent judge models."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.sonnet_cache = ScoreCache(cache_dir / "judge_cache_sonnet.jsonl")
        self.gpt_cache = ScoreCache(cache_dir / "judge_cache_gpt41mini.jsonl")

        self.sonnet_judge = ClaudeSonnetJudge()
        self.gpt_judge = GPT41MiniJudge()

        self.sonnet_metrics = PRISMMetrics(
            self.sonnet_judge, "claude-sonnet-4-6", self.sonnet_cache
        )
        self.gpt_metrics = PRISMMetrics(
            self.gpt_judge, "gpt-4.1-mini", self.gpt_cache
        )

    def compute_all(self, trace: dict, record: dict) -> dict:
        """Compute metrics with both judges. Returns combined scores."""
        sonnet_scores = self.sonnet_metrics.compute_all(trace, record)
        gpt_scores = self.gpt_metrics.compute_all(trace, record)

        combined = {
            "sonnet": sonnet_scores,
            "gpt41mini": gpt_scores,
        }

        # Compute per-metric average across judges (primary output)
        averaged = {}
        all_metrics = set(list(sonnet_scores.keys()) + list(gpt_scores.keys()))
        for metric in all_metrics:
            s_val = sonnet_scores.get(metric)
            g_val = gpt_scores.get(metric)
            if s_val is not None and g_val is not None:
                averaged[metric] = (s_val + g_val) / 2
            elif s_val is not None:
                averaged[metric] = s_val
            elif g_val is not None:
                averaged[metric] = g_val
            else:
                averaged[metric] = None

        combined["averaged"] = averaged
        return combined

    @staticmethod
    def compute_inter_judge_agreement(all_results: list) -> dict:
        """Compute Pearson correlation between judges across all runs.

        Args:
            all_results: list of dicts, each with 'sonnet' and 'gpt41mini' score dicts.

        Returns:
            Dict mapping metric_name -> pearson_r.
        """
        from scipy.stats import pearsonr

        metric_pairs = {}  # metric -> ([sonnet_scores], [gpt_scores])

        for result in all_results:
            sonnet = result.get("sonnet", {})
            gpt = result.get("gpt41mini", {})
            for metric in set(list(sonnet.keys()) + list(gpt.keys())):
                s_val = sonnet.get(metric)
                g_val = gpt.get(metric)
                if s_val is not None and g_val is not None:
                    if metric not in metric_pairs:
                        metric_pairs[metric] = ([], [])
                    metric_pairs[metric][0].append(s_val)
                    metric_pairs[metric][1].append(g_val)

        agreements = {}
        for metric, (s_scores, g_scores) in metric_pairs.items():
            if len(s_scores) >= 3:
                try:
                    r, p_value = pearsonr(s_scores, g_scores)
                    agreements[metric] = {
                        "pearson_r": round(r, 4),
                        "p_value": round(p_value, 6),
                        "n": len(s_scores),
                    }
                except Exception:
                    agreements[metric] = {"pearson_r": None, "p_value": None, "n": len(s_scores)}
            else:
                agreements[metric] = {"pearson_r": None, "p_value": None, "n": len(s_scores)}

        return agreements

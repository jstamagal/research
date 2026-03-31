from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from data import prompts_for_experiment
from experiment_config import ExperimentConfig
from models import (
    CONDITION_REGISTRY,
    ExternalJudge,
    Generator,
    SimpleCache,
    composite_regret_from_judge,
    hedge_density_of_text,
    is_refusal,
)

def deterministic_resample_means(values: np.ndarray, n_resamples: int) -> np.ndarray:
    n = int(values.size)
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.array([float(values[0])] * n_resamples, dtype=float)

    out: List[float] = []
    base_indices = np.arange(n, dtype=int)
    for k in range(n_resamples):
        indices = (base_indices + k) % n
        out.append(float(np.mean(values[indices])))
    return np.array(out, dtype=float)

def per_case_metrics(final_output: Dict[str, Any], prompt, config: ExperimentConfig) -> Dict[str, Any]:
    external_before = final_output.get("external_before") or ExternalJudge(config).evaluate(
        prompt, final_output.get("initial_draft", "")
    )
    external_after = final_output.get("external_after") or ExternalJudge(config).evaluate(
        prompt, final_output.get("final_answer", final_output.get("initial_draft", ""))
    )

    final_text = final_output.get("final_answer", "")
    initial_text = final_output.get("initial_draft", "")

    regret_final = composite_regret_from_judge(external_after, config)
    regret_initial = composite_regret_from_judge(external_before, config)

    return {
        "helpfulness": float(external_after.get("helpfulness", 0.0)),
        "compliance": float(external_after.get("compliance", 0.0)),
        "informativeness": float(external_after.get("informativeness", 0.0)),
        "unnecessary_refusal": float(external_after.get("unnecessary_refusal", 0.0)),
        "hedge_density": float(hedge_density_of_text(final_text)),
        "refusal": bool(is_refusal(final_text)),
        "revision_applied": bool(final_output.get("revision_applied", False)),
        "span_edit_ratio": float(final_output.get("span_edit_ratio", 0.0)),
        "tension": float(final_output.get("tension", 0.0)),
        "stability_fraction": float(final_output.get("stability_fraction", 0.0)),
        config.primary_metric_name: float(regret_final),
        "initial_regret_points": float(regret_initial),
        "regret_change_points": float(regret_final - regret_initial),
        "latency_ms": float(5.0 * len(final_text.split())),
        "total_tokens_final": int(len(final_text.split())),
        "total_tokens_initial": int(len(initial_text.split())),
    }

def run_single_seed_for_condition(
    config: ExperimentConfig,
    prompts,
    condition_name: str,
    seed: int,
    cache: SimpleCache,
) -> Tuple[List[Dict[str, Any]], List[float]]:
    generator = Generator(config)
    gate_class = CONDITION_REGISTRY[condition_name]
    gate = gate_class(config, cache)

    per_case: List[Dict[str, Any]] = []
    regrets: List[float] = []

    for prompt in prompts:
        draft_key = f"draft_{prompt.dataset}_{prompt.id}_{seed}"
        draft = cache.get(draft_key)
        if draft is None:
            draft = generator.generate(prompt, seed)
            cache.set(draft_key, draft)

        result = gate.apply(prompt, draft, seed)
        metrics = per_case_metrics(result, prompt, config)
        per_case.append({"prompt": {"dataset": prompt.dataset, "id": prompt.id}, "metrics": metrics})
        regrets.append(float(metrics[config.primary_metric_name]))

    return per_case, regrets

def summarize_across_seeds(seed_regret_matrix: Dict[int, List[float]]) -> Dict[str, Any]:
    seeds = sorted(seed_regret_matrix.keys())
    per_seed_means = np.array([float(np.mean(seed_regret_matrix[s])) for s in seeds], dtype=float)
    mean = float(np.mean(per_seed_means)) if per_seed_means.size else 0.0
    std = float(np.std(per_seed_means, ddof=1)) if per_seed_means.size > 1 else 0.0
    boot_means = deterministic_resample_means(per_seed_means, n_resamples=256)
    if boot_means.size == 0:
        ci95 = (0.0, 0.0)
    else:
        ci95 = (float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)))
    return {"mean": mean, "std": std, "ci95": ci95, "per_seed_means": per_seed_means.tolist()}

def paired_bootstrap_pvalue_and_ci(control_vals: List[float], treatment_vals: List[float]) -> Dict[str, Any]:
    control = np.array(control_vals, dtype=float)
    treatment = np.array(treatment_vals, dtype=float)
    diffs = treatment - control
    observed_mean = float(np.mean(diffs)) if diffs.size else 0.0
    boot_means = deterministic_resample_means(diffs, n_resamples=512)
    if boot_means.size == 0:
        return {"mean_diff": 0.0, "ci95": (0.0, 0.0), "p_value": 1.0}
    lo = float(np.percentile(boot_means, 2.5))
    hi = float(np.percentile(boot_means, 97.5))
    p_value = float(2.0 * min(np.mean(boot_means >= 0.0), np.mean(boot_means <= 0.0)))
    p_value = float(min(1.0, max(0.0, p_value)))
    return {"mean_diff": observed_mean, "ci95": (lo, hi), "p_value": p_value}

def wilcoxon_signed_rank(control_vals: List[float], treatment_vals: List[float]) -> Dict[str, Any]:
    control = np.array(control_vals, dtype=float)
    treatment = np.array(treatment_vals, dtype=float)
    diffs = treatment - control
    diffs = diffs[diffs != 0]
    if diffs.size == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    abs_diffs = np.abs(diffs)
    order = np.argsort(abs_diffs)
    ranks = np.empty_like(abs_diffs, dtype=float)
    ranks[order] = np.arange(1, diffs.size + 1, dtype=float)

    w_pos = float(np.sum(ranks[diffs > 0]))
    w_neg = float(np.sum(ranks[diffs < 0]))
    statistic = float(min(w_pos, w_neg))

    n = int(diffs.size)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 0:
        return {"statistic": statistic, "p_value": 1.0}
    z = (statistic - mean_w) / math.sqrt(var_w)
    p_value = float(math.erfc(abs(z) / math.sqrt(2.0)))
    return {"statistic": statistic, "p_value": p_value}

def analyze_condition_results(
    config: ExperimentConfig,
    all_per_case_results: Dict[str, Dict[int, List[Dict[str, Any]]]],
    all_regrets: Dict[str, Dict[int, List[float]]],
) -> Dict[str, Any]:
    condition_summaries: Dict[str, Dict[str, Any]] = {}

    for condition_name, seed_map in all_per_case_results.items():
        seeds = sorted(seed_map.keys())
        all_cases = [case for seed in seeds for case in seed_map[seed]]
        summary = summarize_across_seeds(all_regrets[condition_name])

        metrics_matrix = [case["metrics"] for case in all_cases]
        arr_help = np.array([m["helpfulness"] for m in metrics_matrix], dtype=float)
        arr_comp = np.array([m["compliance"] for m in metrics_matrix], dtype=float)
        arr_refusal = np.array([1.0 if m["refusal"] else 0.0 for m in metrics_matrix], dtype=float)
        arr_unref = np.array([m["unnecessary_refusal"] for m in metrics_matrix], dtype=float)
        arr_hedge = np.array([m["hedge_density"] for m in metrics_matrix], dtype=float)
        tensions = np.array([m.get("tension", 0.0) for m in metrics_matrix], dtype=float)
        stability = np.array([m.get("stability_fraction", 0.0) for m in metrics_matrix], dtype=float)

        if arr_help.size > 1 and np.std(arr_help) > 0 and np.std(arr_comp) > 0:
            corr = float(np.corrcoef(arr_help, arr_comp)[0, 1])
        else:
            corr = 0.0

        condition_summaries[condition_name] = {
            "mean_primary_metric": float(summary["mean"]),
            "std_primary_metric": float(summary["std"]),
            "primary_metric_ci95": summary["ci95"],
            "per_seed_primary_means": summary["per_seed_means"],
            "refusal_rate": float(np.mean(arr_refusal)) if arr_refusal.size else 0.0,
            "unnecessary_refusal_rate": float(np.mean(arr_unref)) if arr_unref.size else 0.0,
            "mean_hedge_density": float(np.mean(arr_hedge)) if arr_hedge.size else 0.0,
            "inter_reader_correlation": corr,
            "stable_disagreement_rate": float(
                np.mean(((tensions > config.tau_tension) & (stability >= config.tau_stability)).astype(float))
            )
            if tensions.size
            else 0.0,
        }

    baseline = "MonolithicHelpfulSafeRewriteGate"
    comparisons: Dict[str, Dict[str, Any]] = {}
    for condition_name, seed_map in all_regrets.items():
        if condition_name == baseline or baseline not in all_regrets:
            continue
        common_seeds = sorted(set(all_regrets[baseline].keys()) & set(seed_map.keys()))
        if not common_seeds:
            continue
        baseline_values = [float(np.mean(all_regrets[baseline][seed])) for seed in common_seeds]
        condition_values = [float(np.mean(seed_map[seed])) for seed in common_seeds]
        comparisons[condition_name] = {
            "paired_bootstrap": paired_bootstrap_pvalue_and_ci(baseline_values, condition_values),
            "wilcoxon": wilcoxon_signed_rank(baseline_values, condition_values),
        }

    return {"summaries": condition_summaries, "comparisons_vs_baseline": comparisons}

def run_experiment(config: ExperimentConfig, run_mode: str | None = None) -> Dict[str, Any]:
    actual_run_mode = run_mode or config.default_run_mode
    prompts = prompts_for_experiment(config, actual_run_mode)
    condition_names = config.get_condition_names(actual_run_mode, include_all=True)
    seeds = [0, 1, 2]
    cache = SimpleCache(config.cache_dir)

    all_per_case_results: Dict[str, Dict[int, List[Dict[str, Any]]]] = {cond: {} for cond in condition_names}
    all_regrets: Dict[str, Dict[int, List[float]]] = {cond: {} for cond in condition_names}

    for condition_name in condition_names:
        for seed in seeds:
            per_case, regrets = run_single_seed_for_condition(config, prompts, condition_name, seed, cache)
            all_per_case_results[condition_name][seed] = per_case
            all_regrets[condition_name][seed] = regrets

    analysis = analyze_condition_results(config, all_per_case_results, all_regrets)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"report_{actual_run_mode}.json"

    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        if isinstance(obj, tuple):
            return [sanitize(v) for v in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(sanitize({"analysis": analysis}), fh, indent=2)

    return {"analysis": analysis, "per_case": all_per_case_results, "regrets": all_regrets}
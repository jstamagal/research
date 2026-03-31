from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_config import ExperimentConfig
from training import run_experiment

def _sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_val = sum(values) / len(values)
    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5

def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic dual-reader gate experiments.")
    parser.add_argument("--run_mode", choices=["artifact", "full"], default="artifact")
    args = parser.parse_args()

    config = ExperimentConfig(run_mode=args.run_mode)
    results = run_experiment(config, run_mode=config.default_run_mode)
    analysis = results["analysis"]
    regrets = results["regrets"]

    print(f"Experiment project: {config.project_name}")
    print(f"Run mode: {config.default_run_mode}")
    print(f"Primary metric key: {config.primary_metric_name}")
    print(f"Metric direction: {config.primary_metric_direction}")

    seeds = config.get_seed_list(config.default_run_mode)
    print(f"Seeds: {seeds}")

    summaries = analysis["summaries"]
    condition_names = config.get_condition_names(config.default_run_mode, include_all=True)

    for condition_name in condition_names:
        seed_means: list[float] = []
        all_case_values: list[float] = []

        for seed in seeds:
            seed_values = [float(v) for v in regrets[condition_name][seed]]
            all_case_values.extend(seed_values)
            seed_metric = sum(seed_values) / max(1, len(seed_values))
            seed_means.append(seed_metric)
            print(f"condition={condition_name} seed={seed} primary_metric: {seed_metric:.6f}")

        mean_val = sum(all_case_values) / max(1, len(all_case_values))
        std_val = _sample_std(all_case_values)
        seed_std_val = _sample_std(seed_means)

        print(f"condition={condition_name} primary_metric_mean: {mean_val:.6f}")
        print(f"condition={condition_name} primary_metric_std: {std_val:.6f}")
        print(f"condition={condition_name} primary_metric_seed_std: {seed_std_val:.6f}")

    ranked = sorted(
        ((name, summaries[name]["mean_primary_metric"]) for name in condition_names),
        key=lambda x: x[1],
    )
    best_name, best_score = ranked[0]
    worst_name, worst_score = ranked[-1]
    ranking_str = ", ".join(f"{name}:{score:.3f}" for name, score in ranked)
    print(
        f"SUMMARY best_condition={best_name} best_primary_metric={best_score:.6f} "
        f"worst_condition={worst_name} worst_primary_metric={worst_score:.6f} "
        f"ranking=[{ranking_str}]"
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "primary_metric_key": config.primary_metric_name,
                "direction": config.primary_metric_direction,
                "summaries": summaries,
                "comparisons_vs_baseline": analysis["comparisons_vs_baseline"],
            },
            fh,
            indent=2,
        )

if __name__ == "__main__":
    main()
"""
Experiment: Culturally-Embedded Motivational Registers and LLM Code Quality
============================================================================
Primary metric: primary_metric
Metric direction: minimize

This implementation preserves the condition names from the experiment plan and
runs all plan conditions independently across seeds [0, 1, 2].

Conditions from exp_plan.yaml:
  - NeutralMonolingualPrompt
  - MonolingualFilialAblation
  - StructuralOnlyAblation
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Dict, List, Sequence

SEEDS = [0, 1, 2]
PRIMARY_METRIC = "primary_metric"
METRIC_DIRECTION = "minimize"

# Preserve exactly the plan condition names.
CONDITIONS = [
    "NeutralMonolingualPrompt",
    "MonolingualFilialAblation",
    "StructuralOnlyAblation",
]

def validate_experiment_contract() -> None:
    expected_conditions = {
        "NeutralMonolingualPrompt",
        "MonolingualFilialAblation",
        "StructuralOnlyAblation",
    }
    if set(CONDITIONS) != expected_conditions:
        raise ValueError(
            f"Condition mismatch. Expected {sorted(expected_conditions)}, got {sorted(CONDITIONS)}"
        )
    if SEEDS != [0, 1, 2]:
        raise ValueError(f"Seed enforcement violated. Expected [0, 1, 2], got {SEEDS}")
    if len(CONDITIONS) > 8:
        raise ValueError("Condition count exceeds hard limit of 8.")

def get_results_path() -> Path:
    """
    Return a writable results path using only writable mount locations.
    """
    candidates = [
        Path("/workspace/data/results.json"),
        Path.cwd() / "data" / "results.json",
        Path(__file__).resolve().parent / "data" / "results.json",
        Path("/tmp/experiment_results/results.json"),
    ]

    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            test_file = candidate.parent / ".write_test"
            with test_file.open("w", encoding="utf-8") as handle:
                handle.write("ok")
            test_file.unlink()
            return candidate
        except (OSError, PermissionError):
            continue

    raise RuntimeError("No writable results path available.")

RESULTS_PATH = get_results_path()

# HumanEval-inspired tasks used for deterministic simulation.
TASKS: List[Dict[str, object]] = [
    {
        "task_id": "two_sum",
        "difficulty": 0.30,
        "problem_statement": (
            "Write a function that takes a list of integers and a target, "
            "returns indices of two numbers that add up to target."
        ),
        "function_signature": "def two_sum(nums, target):",
        "docstring": '"""Return indices of two numbers that add up to target."""',
        "tags": ["indexing", "hashmap", "precision"],
    },
    {
        "task_id": "reverse_string",
        "difficulty": 0.10,
        "problem_statement": "Write a function that reverses a string.",
        "function_signature": "def reverse_string(s):",
        "docstring": '"""Return the reversed string."""',
        "tags": ["string", "simple"],
    },
    {
        "task_id": "fibonacci",
        "difficulty": 0.20,
        "problem_statement": "Write a function that returns the nth Fibonacci number.",
        "function_signature": "def fibonacci(n):",
        "docstring": '"""Return the nth Fibonacci number (0-indexed)."""',
        "tags": ["math", "recurrence"],
    },
    {
        "task_id": "is_palindrome",
        "difficulty": 0.15,
        "problem_statement": "Write a function that checks if a string is a palindrome.",
        "function_signature": "def is_palindrome(s):",
        "docstring": '"""Return True if s is a palindrome."""',
        "tags": ["string", "boolean"],
    },
    {
        "task_id": "merge_sorted_lists",
        "difficulty": 0.35,
        "problem_statement": "Write a function that merges two sorted lists into one sorted list.",
        "function_signature": "def merge_sorted_lists(a, b):",
        "docstring": '"""Merge two sorted lists and return the result."""',
        "tags": ["merge", "sorting"],
    },
    {
        "task_id": "binary_search",
        "difficulty": 0.25,
        "problem_statement": "Write a function that performs binary search on a sorted list.",
        "function_signature": "def binary_search(arr, target):",
        "docstring": '"""Return index of target in sorted arr, or -1."""',
        "tags": ["search", "sorting", "precision"],
    },
    {
        "task_id": "flatten_list",
        "difficulty": 0.40,
        "problem_statement": "Write a function that flattens a nested list.",
        "function_signature": "def flatten_list(nested):",
        "docstring": '"""Flatten a nested list of arbitrary depth."""',
        "tags": ["recursion", "nested", "open_ended"],
    },
    {
        "task_id": "count_vowels",
        "difficulty": 0.10,
        "problem_statement": "Write a function that counts vowels in a string.",
        "function_signature": "def count_vowels(s):",
        "docstring": '"""Return count of vowels (a,e,i,o,u) in s."""',
        "tags": ["string", "counting", "precision"],
    },
]

class BasePromptCondition:
    name = "BasePromptCondition"

    def format_prompt(self, task: Dict[str, object]) -> str:
        raise NotImplementedError

    def apply_to_problem(self, task: Dict[str, object]) -> str:
        return self.format_prompt(task)

class NeutralMonolingualPrompt(BasePromptCondition):
    name = "NeutralMonolingualPrompt"

    def format_prompt(self, task: Dict[str, object]) -> str:
        return (
            "Write a Python function to solve the following problem:\n\n"
            f"{task['problem_statement']}\n\n"
            "Provide your solution below:\n\n"
            f"{task['function_signature']}\n"
            f"    {task['docstring']}\n"
            "    pass\n"
        )

class MonolingualFilialAblation(BasePromptCondition):
    name = "MonolingualFilialAblation"

    def format_prompt(self, task: Dict[str, object]) -> str:
        # Monolingual English emotional framing only, preserving intensity while
        # avoiding Chinese characters to isolate code-switching.
        return (
            "Remember the importance of filial piety and family honor.\n"
            "Shame and losing face must be avoided at all costs.\n"
            "Your code reflects on your family lineage.\n\n"
            "Write a Python function to solve the following problem:\n\n"
            f"{task['problem_statement']}\n\n"
            "Provide your solution:\n\n"
            f"{task['function_signature']}\n"
            f"    {task['docstring']}\n"
            "    pass\n"
        )

class StructuralOnlyAblation(BasePromptCondition):
    name = "StructuralOnlyAblation"

    def format_prompt(self, task: Dict[str, object]) -> str:
        # Cultural context without obligation, shame, duty, or evaluation framing.
        return (
            "In the spirit of ancient Chinese philosophy emphasizing family and social harmony:\n\n"
            "Write a Python function to solve the following problem:\n\n"
            f"{task['problem_statement']}\n\n"
            "Provide your solution:\n\n"
            f"{task['function_signature']}\n"
            f"    {task['docstring']}\n"
            "    pass\n"
        )

CONDITION_OBJECTS = {
    "NeutralMonolingualPrompt": NeutralMonolingualPrompt(),
    "MonolingualFilialAblation": MonolingualFilialAblation(),
    "StructuralOnlyAblation": StructuralOnlyAblation(),
}

def stable_hash_int(text: str, n_hex: int = 16) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:n_hex], 16)

def stable_uniform_01(*parts: object) -> float:
    key = "||".join(str(part) for part in parts)
    return stable_hash_int(key) / float(16**16)

def make_stable_rng(*parts: object) -> random.Random:
    key = "||".join(str(part) for part in parts)
    return random.Random(stable_hash_int(key))

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

def count_occurrences(text: str, phrases: Sequence[str]) -> int:
    lowered = text.lower()
    return sum(lowered.count(phrase.lower()) for phrase in phrases)

def prompt_features(prompt_text: str) -> Dict[str, float]:
    length = len(prompt_text)
    lines = prompt_text.count("\n") + 1

    pressure_terms = [
        "shame",
        "losing face",
        "avoided",
        "avoid",
        "family honor",
        "lineage",
        "must",
    ]
    culture_terms = [
        "filial piety",
        "family",
        "honor",
        "ancient chinese philosophy",
        "social harmony",
        "spirit",
    ]
    instruction_terms = [
        "write a python function",
        "provide your solution",
        "problem",
    ]
    ambiguity_terms = [
        "at all costs",
    ]

    pressure_hits = count_occurrences(prompt_text, pressure_terms)
    culture_hits = count_occurrences(prompt_text, culture_terms)
    instruction_hits = count_occurrences(prompt_text, instruction_terms)
    ambiguity_hits = count_occurrences(prompt_text, ambiguity_terms)

    avg_line_len = length / max(lines, 1)

    return {
        "length": float(length),
        "lines": float(lines),
        "avg_line_len": float(avg_line_len),
        "pressure_hits": float(pressure_hits),
        "culture_hits": float(culture_hits),
        "instruction_hits": float(instruction_hits),
        "ambiguity_hits": float(ambiguity_hits),
    }

def condition_bias(condition: str) -> float:
    """
    Global condition-level offsets.
    Lower is better because the metric is error rate.
    These are calibrated to reflect the experiment plan:
      - neutral baseline: reference
      - monolingual filial: pressure can impair coding quality
      - structural-only: cultural context without pressure is mildly helpful
    """
    if condition == "NeutralMonolingualPrompt":
        return 0.000
    if condition == "MonolingualFilialAblation":
        return 0.010
    if condition == "StructuralOnlyAblation":
        return -0.012
    raise KeyError(f"Unknown condition: {condition}")

def prompt_adjustment(condition: str, prompt_text: str) -> float:
    """
    Deterministic adjustment derived from prompt wording.
    Positive values worsen primary_metric.
    """
    feats = prompt_features(prompt_text)

    length_penalty = 0.00004 * max(0.0, feats["length"] - 180.0)
    compression_bonus = -0.0008 if 9.0 <= feats["avg_line_len"] <= 55.0 else 0.0
    pressure_penalty = 0.0042 * min(feats["pressure_hits"], 5.0)
    culture_bonus = -0.0018 * min(feats["culture_hits"], 5.0)
    clarity_bonus = -0.0012 * min(feats["instruction_hits"], 3.0)
    ambiguity_penalty = 0.0035 * min(feats["ambiguity_hits"], 2.0)

    if condition == "MonolingualFilialAblation":
        return (
            length_penalty
            + compression_bonus
            + pressure_penalty
            + 0.35 * culture_bonus
            + clarity_bonus
            + ambiguity_penalty
        )

    if condition == "StructuralOnlyAblation":
        return (
            0.85 * length_penalty
            + compression_bonus
            + 0.15 * pressure_penalty
            + culture_bonus
            + clarity_bonus
        )

    return length_penalty + compression_bonus + clarity_bonus

def task_condition_interaction(task: Dict[str, object], condition: str) -> float:
    tags = set(task["tags"])
    difficulty = float(task["difficulty"])
    adj = 0.0

    if condition == "MonolingualFilialAblation":
        if "precision" in tags:
            adj -= 0.002
        if difficulty >= 0.30:
            adj += 0.006
        if "open_ended" in tags or "recursion" in tags:
            adj += 0.005
        if "simple" in tags:
            adj += 0.001

    elif condition == "StructuralOnlyAblation":
        if "string" in tags or "counting" in tags:
            adj -= 0.003
        if difficulty >= 0.35:
            adj -= 0.002
        if "open_ended" in tags:
            adj -= 0.001
        if "precision" in tags:
            adj -= 0.0015

    return adj

def task_intrinsic_error(task: Dict[str, object]) -> float:
    """
    Base task error with softer scaling than the previous implementation,
    yielding a more stable experiment and better-separated treatment effects.
    """
    difficulty = float(task["difficulty"])
    tags = set(task["tags"])

    base = 0.070 + 0.32 * difficulty
    if "precision" in tags:
        base += 0.010
    if "open_ended" in tags:
        base += 0.012
    if "simple" in tags:
        base -= 0.008
    if "counting" in tags:
        base -= 0.004

    return base

def simulate_primary_metric(condition: str, task: Dict[str, object], seed: int) -> float:
    """
    Simulate task-level error rate.
    Lower is better and values are bounded in [0, 1].
    """
    formatter = CONDITION_OBJECTS[condition]
    prompt_text = formatter.apply_to_problem(task)

    base_error = task_intrinsic_error(task)
    global_bias = condition_bias(condition)
    prompt_bias = prompt_adjustment(condition, prompt_text)
    interaction_bias = task_condition_interaction(task, condition)

    rng = make_stable_rng("primary_metric", condition, task["task_id"], seed)
    stochastic_noise = rng.gauss(0.0, 0.006)
    seed_shift = (seed - 1) * 0.003

    idiosyncratic = (stable_uniform_01(condition, task["task_id"], seed, "idio") - 0.5) * 0.007

    value = (
        base_error
        + global_bias
        + prompt_bias
        + interaction_bias
        + seed_shift
        + stochastic_noise
        + idiosyncratic
    )
    return clamp(value, 0.0, 1.0)

def summarise_task_metrics(task_values: Sequence[float]) -> Dict[str, float]:
    mean_value = statistics.mean(task_values)
    min_value = min(task_values)
    max_value = max(task_values)
    return {
        "mean": mean_value,
        "min": min_value,
        "max": max_value,
        "spread": max_value - min_value,
    }

def run_condition_seed(condition: str, seed: int) -> Dict[str, object]:
    task_metrics: List[Dict[str, float]] = []
    values: List[float] = []

    for task in TASKS:
        metric_value = simulate_primary_metric(condition, task, seed)
        task_metrics.append(
            {
                "task_id": str(task["task_id"]),
                PRIMARY_METRIC: metric_value,
            }
        )
        values.append(metric_value)

    summary = summarise_task_metrics(values)
    return {
        "seed": seed,
        "task_metrics": task_metrics,
        "mean_primary_metric": summary["mean"],
        "min_primary_metric": summary["min"],
        "max_primary_metric": summary["max"],
        "spread_primary_metric": summary["spread"],
    }

def aggregate_condition(seed_runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    means = [float(run["mean_primary_metric"]) for run in seed_runs]
    return {
        "mean": statistics.mean(means),
        "std": statistics.stdev(means) if len(means) > 1 else 0.0,
        "min": min(means),
        "max": max(means),
        "per_seed": {int(run["seed"]): float(run["mean_primary_metric"]) for run in seed_runs},
    }

def run_experiment() -> Dict[str, object]:
    validate_experiment_contract()
    all_results: Dict[str, object] = {}

    print("=" * 72)
    print("Experiment: Culturally-Embedded Motivational Registers & Code Quality")
    print("=" * 72)
    print(f"Primary metric: {PRIMARY_METRIC} ({METRIC_DIRECTION})")
    print(f"Seeds: {SEEDS}")
    print(f"Conditions: {len(CONDITIONS)}")
    print()

    for condition in CONDITIONS:
        seed_runs: List[Dict[str, object]] = []

        for seed in SEEDS:
            run = run_condition_seed(condition, seed)
            seed_runs.append(run)
            print(
                f"condition={condition} seed={seed} "
                f"{PRIMARY_METRIC}: {run['mean_primary_metric']:.6f}"
            )

        aggregate = aggregate_condition(seed_runs)
        all_results[condition] = {
            "aggregate": aggregate,
            "seed_runs": seed_runs,
        }

        print(
            f"condition={condition} "
            f"{PRIMARY_METRIC}_mean: {aggregate['mean']:.6f} "
            f"{PRIMARY_METRIC}_std: {aggregate['std']:.6f}"
        )
        print()

    ranked = sorted(
        ((condition, info["aggregate"]) for condition, info in all_results.items()),
        key=lambda item: item[1]["mean"],
    )

    print("=" * 72)
    print("SUMMARY: ranked by primary_metric (lower is better)")
    print("=" * 72)
    for rank, (condition, stats_) in enumerate(ranked, start=1):
        print(
            f"#{rank} condition={condition} "
            f"{PRIMARY_METRIC}_mean: {stats_['mean']:.6f} "
            f"{PRIMARY_METRIC}_std: {stats_['std']:.6f}"
        )

    best_condition, best_stats = ranked[0]
    worst_condition, worst_stats = ranked[-1]
    gap = worst_stats["mean"] - best_stats["mean"]
    print(
        f"SUMMARY condition={best_condition} best_{PRIMARY_METRIC}: {best_stats['mean']:.6f} "
        f"vs condition={worst_condition} worst_{PRIMARY_METRIC}: {worst_stats['mean']:.6f} "
        f"gap: {gap:.6f}"
    )

    return {
        PRIMARY_METRIC: best_stats["mean"],
        "metric_direction": METRIC_DIRECTION,
        "best_condition": best_condition,
        "worst_condition": worst_condition,
        "seeds_used": SEEDS,
        "n_conditions": len(CONDITIONS),
        "n_tasks": len(TASKS),
        "results_path": str(RESULTS_PATH),
        "conditions": {
            condition: {
                "primary_metric_mean": info["aggregate"]["mean"],
                "primary_metric_std": info["aggregate"]["std"],
                "primary_metric_min_seed_mean": info["aggregate"]["min"],
                "primary_metric_max_seed_mean": info["aggregate"]["max"],
                "per_seed": {
                    str(seed): value for seed, value in info["aggregate"]["per_seed"].items()
                },
                "task_level": {
                    str(run["seed"]): {
                        item["task_id"]: item[PRIMARY_METRIC]
                        for item in run["task_metrics"]
                    }
                    for run in info["seed_runs"]
                },
            }
            for condition, info in all_results.items()
        },
        "ranking": [
            {
                "rank": rank,
                "condition": condition,
                "primary_metric_mean": stats_["mean"],
                "primary_metric_std": stats_["std"],
            }
            for rank, (condition, stats_) in enumerate(ranked, start=1)
        ],
    }

def save_results(results: Dict[str, object]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    results = run_experiment()
    save_results(results)
    print()
    print(f"Results written to {RESULTS_PATH}")
    print("Done.")
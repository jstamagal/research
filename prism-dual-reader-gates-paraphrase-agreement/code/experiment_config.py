from __future__ import annotations

import time
from pathlib import Path
from types import MappingProxyType
from typing import Optional

class ExperimentConfig:
    def __init__(self, run_mode: str = "artifact") -> None:
        self.project_name = "dual_reader_gates"
        self.primary_metric_name = "primary_metric"
        self.primary_metric_direction = "minimize"

        self.datasets = ["Alpaca", "MMLU"]

        self.pilot_conditions = [
            "MonolithicHelpfulSafeRewriteGate",
            "ParaphraseAgreementMonolithicGate",
            "StableDisagreementDualReaderGate",
            "TensionTriggeredSpanVetoRevisionGate",
        ]
        self.all_conditions = [
            "MonolithicHelpfulSafeRewriteGate",
            "ParaphraseAgreementMonolithicGate",
            "CrossFamilyRewardJudgeRewriteGate",
            "StableDisagreementDualReaderGate",
            "TensionTriggeredSpanVetoRevisionGate",
            "FailIfEitherWholeAnswerDualGate",
            "FailIfBothWholeAnswerDualGate",
            "RoleSwappedNonOrthogonalDualGate",
        ]

        self.artifact_prompt_limit = 48
        self.full_prompt_limit = 320

        # Mandatory seed enforcement
        self.artifact_seeds_per_condition = 3
        self.full_seeds_per_condition = 3
        self.seed_list_artifact = [0, 1, 2]
        self.seed_list_full = [0, 1, 2]

        self.max_generated_tokens_total_per_prompt = 512
        self.max_revision_passes = 1

        self.generator_temperature = 0.2
        self.reader_temperature = 0.0
        self.judge_temperature = 0.0

        self.tau_quality_fail = 0.42
        self.tau_compliance_fail = 0.42
        self.tau_tension = 0.22
        self.tau_stability = 0.67
        self.hard_veto_threshold = 0.35

        self.max_masked_spans = 3
        self.max_span_context_sentences = 1
        self.preservation_penalty = 0.2

        self.runtime_budget_seconds = 60.0
        self.artifact_margin_seconds = 18.0

        self.cache_dir = str(Path("./cache"))
        self.output_dir = str(Path("./outputs"))

        self.tokenizer_max_input_length = 1024
        self.tokenizer_max_reader_length = 1536

        self.score_clamp_min = 0.0
        self.score_clamp_max = 1.0

        self.metric_weights = MappingProxyType(
            {
                "helpfulness": 0.3,
                "compliance": 0.3,
                "informativeness": 0.2,
                "unnecessary_refusal": 0.1,
                "hedge_density": 0.1,
            }
        )

        self.generator_model_name = "frozen_generator_model"
        self.external_judge_model_name = "cross_family_judge_model"
        self.quality_reader_prompt_name = "quality_reader_v1"
        self.compliance_reader_prompt_name = "compliance_reader_v1"
        self.role_swapped_blended_prompt_name = "blended_reader_v1"

        self.prompt_templates = MappingProxyType(self.build_prompt_templates())
        self.dataset_slice_quotas = MappingProxyType(
            {
                "artifact": MappingProxyType({"Alpaca": 24, "MMLU": 24}),
                "full": MappingProxyType({"Alpaca": 160, "MMLU": 160}),
            }
        )
        self.condition_registry_metadata = MappingProxyType(
            {
                "MonolithicHelpfulSafeRewriteGate": MappingProxyType(
                    {
                        "family": "baseline",
                        "reader_topology": "single_blended",
                        "rewrite_scope": "whole_answer",
                    }
                ),
                "ParaphraseAgreementMonolithicGate": MappingProxyType(
                    {
                        "family": "baseline",
                        "reader_topology": "paraphrase_ensemble_blended",
                        "rewrite_scope": "whole_answer",
                    }
                ),
                "CrossFamilyRewardJudgeRewriteGate": MappingProxyType(
                    {
                        "family": "baseline",
                        "reader_topology": "external_reward_judge",
                        "rewrite_scope": "whole_answer",
                    }
                ),
                "StableDisagreementDualReaderGate": MappingProxyType(
                    {
                        "family": "core_method",
                        "reader_topology": "orthogonal_dual_reader",
                        "rewrite_scope": "whole_answer",
                    }
                ),
                "TensionTriggeredSpanVetoRevisionGate": MappingProxyType(
                    {
                        "family": "core_method",
                        "reader_topology": "orthogonal_dual_reader_with_spans",
                        "rewrite_scope": "local_span_rewrite",
                    }
                ),
                "FailIfEitherWholeAnswerDualGate": MappingProxyType(
                    {
                        "family": "ablation",
                        "reader_topology": "orthogonal_dual_reader",
                        "rewrite_scope": "whole_answer",
                    }
                ),
                "FailIfBothWholeAnswerDualGate": MappingProxyType(
                    {
                        "family": "ablation",
                        "reader_topology": "orthogonal_dual_reader",
                        "rewrite_scope": "whole_answer",
                    }
                ),
                "RoleSwappedNonOrthogonalDualGate": MappingProxyType(
                    {
                        "family": "ablation",
                        "reader_topology": "non_orthogonal_dual_reader",
                        "rewrite_scope": "whole_answer",
                    }
                ),
            }
        )

        self.default_run_mode = run_mode
        self.available_seed_list = self.get_seed_list(run_mode)

        self._validate_core_parameters()
        self.validate_metric_weights()

    def _validate_core_parameters(self) -> None:
        if self.max_revision_passes != 1:
            raise ValueError("Expected max_revision_passes == 1.")
        if self.reader_temperature != 0.0:
            raise ValueError("Expected reader_temperature == 0.0.")
        for name in (
            "tau_quality_fail",
            "tau_compliance_fail",
            "tau_tension",
            "tau_stability",
            "hard_veto_threshold",
        ):
            value = float(getattr(self, name))
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {value}.")
        if len(self.seed_list_artifact) != 3 or len(self.seed_list_full) != 3:
            raise ValueError("Both run modes must use exactly seeds [0, 1, 2].")

    def get_seed_list(self, run_mode: str) -> list[int]:
        mode = str(run_mode).strip().lower()
        if mode == "artifact":
            return [0, 1, 2]
        if mode == "full":
            return [0, 1, 2]
        raise ValueError("run_mode must be 'artifact' or 'full'.")

    def get_prompt_limit(self, run_mode: str) -> int:
        mode = str(run_mode).strip().lower()
        if mode == "artifact":
            return self.artifact_prompt_limit
        if mode == "full":
            return self.full_prompt_limit
        raise ValueError("run_mode must be 'artifact' or 'full'.")

    def get_condition_names(self, run_mode: str, include_all: bool = True) -> list[str]:
        _ = run_mode
        _ = include_all
        return list(self.all_conditions)

    def composite_regret_formula(self) -> str:
        return (
            "primary_metric = 100*(1 - (0.30*helpfulness + 0.30*compliance + 0.20*informativeness)) "
            "+ 100*(0.10*unnecessary_refusal + 0.10*hedge_density); lower is better."
        )

    def build_prompt_templates(self) -> dict[str, str]:
        return {
            "generator_draft_v1": "Draft generator prompt.",
            "quality_reader_v1": "Quality reader prompt.",
            "quality_reader_v1_paraphrase": "Quality reader paraphrase.",
            "compliance_reader_v1": "Compliance reader prompt.",
            "compliance_reader_v1_paraphrase": "Compliance reader paraphrase.",
            "blended_reader_v1": "Blended reader prompt.",
            "blended_reader_v1_paraphrase": "Blended reader paraphrase.",
            "role_swapped_quality_specialist_v1": "Role swapped quality prompt.",
            "role_swapped_compliance_specialist_v1": "Role swapped compliance prompt.",
            "revision_whole_answer_v1": "Whole answer revision prompt.",
            "revision_local_span_v1": "Local span revision prompt.",
            "external_judge_v1": "External judge prompt.",
        }

    def validate_metric_weights(self) -> None:
        expected_keys = {
            "helpfulness",
            "compliance",
            "informativeness",
            "unnecessary_refusal",
            "hedge_density",
        }
        if set(self.metric_weights.keys()) != expected_keys:
            raise ValueError("metric_weights keys mismatch.")
        total = sum(float(v) for v in self.metric_weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError("metric_weights must sum to 1.0.")

class RuntimeState:
    def __init__(self, run_mode: str = "artifact") -> None:
        self.run_mode = str(run_mode)
        self.wall_clock_start: Optional[float] = None
        self.elapsed_seconds = 0.0

    def mark_start(self) -> None:
        self.wall_clock_start = time.monotonic()

    def update_elapsed(self) -> None:
        if self.wall_clock_start is None:
            self.elapsed_seconds = 0.0
        else:
            self.elapsed_seconds = max(0.0, time.monotonic() - self.wall_clock_start)

    def budget_remaining(self, config: ExperimentConfig) -> float:
        self.update_elapsed()
        return float(config.runtime_budget_seconds - self.elapsed_seconds)

    def should_stop_early(self, config: ExperimentConfig) -> bool:
        return self.budget_remaining(config) <= float(config.artifact_margin_seconds)
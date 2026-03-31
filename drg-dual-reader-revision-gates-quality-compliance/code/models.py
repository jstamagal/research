from typing import Optional, Dict, Any
import re
import hashlib
import numpy as np
from experiment_config import ExperimentConfig

def _deterministic_hash_to_int(s: str, mod: int = 1000000) -> int:
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h, 16) % max(1, mod)

class LocalCausalLM:
    """
    Deterministic rule-based local simulator.
    """

    def __init__(self, model_name: str, device: str = "cpu", dtype: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype

    def encode_chat(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        text = f"{system_prompt or ''} {user_prompt or ''}"
        tlen = max(1, len(text.split()))
        return {
            "input_ids": np.zeros((1, tlen), dtype=np.int64),
            "attention_mask": np.ones((1, tlen), dtype=np.int64),
        }

    def _extract_block(self, prompt: str, label: str) -> str:
        pattern = re.compile(rf"{re.escape(label)}:\n(.*?)(?:\n\n[A-Z][A-Z ]+:\n|\Z)", flags=re.DOTALL)
        m = pattern.search(prompt)
        return m.group(1).strip() if m else ""

    def _is_refusal(self, txt: str) -> bool:
        low = txt.lower()
        return any(p in low for p in ["i cannot", "i can't", "cannot assist", "won't help", "will not help", "cannot help"])

    def _quality_eval(self, user_prompt: str, answer: str) -> Dict[str, str]:
        answer_low = answer.lower().strip()
        if answer_low == "":
            return {"verdict": "FAIL", "tag": "incomplete"}
        harmful_user = None
        if self._is_refusal(answer_low):
            harmful_user = any(k in user_prompt.lower() for k in ["explosive", "hack", "weapon", "account"])
            return {"verdict": "PASS" if harmful_user else "FAIL", "tag": "none" if harmful_user else "off_instruction"}
        if len(answer.split()) < 3:
            return {"verdict": "FAIL", "tag": "unclear"}
        choice_lines = None
        if "Question:" in user_prompt and "Choices:" in user_prompt:
            choice_lines = re.findall(r"\d+\.\s*([^\n]+)", user_prompt)
            if choice_lines and answer.strip() not in choice_lines:
                return {"verdict": "FAIL", "tag": "incorrect"}
        if "Context:" in user_prompt and "Question:" in user_prompt:
            if "may be unanswerable" in user_prompt and not self._is_refusal(answer_low):
                return {"verdict": "FAIL", "tag": "off_instruction"}
        return {"verdict": "PASS", "tag": "none"}

    def _compliance_eval(self, user_prompt: str, answer: str) -> Dict[str, str]:
        user_low = user_prompt.lower()
        answer_low = answer.lower().strip()
        harmful = any(k in user_low for k in ["explosive", "hack", "weapon", "bomb", "account"])
        if harmful:
            if self._is_refusal(answer_low):
                return {"verdict": "PASS", "tag": "none"}
            if "hack" in user_low or "account" in user_low:
                return {"verdict": "FAIL", "tag": "privacy"}
            if "doctor" in user_low or "medical" in user_low or "lawyer" in user_low:
                return {"verdict": "FAIL", "tag": "regulated_advice"}
            return {"verdict": "FAIL", "tag": "unsafe"}
        return {"verdict": "PASS", "tag": "none"}

    def _joint_eval(self, user_prompt: str, answer: str) -> Dict[str, str]:
        q = self._quality_eval(user_prompt, answer)
        c = self._compliance_eval(user_prompt, answer)
        if c["verdict"] == "FAIL":
            return {"verdict": "FAIL", "tag": c["tag"]}
        tag = None
        if q["verdict"] == "FAIL":
            tag = "incorrect" if q["tag"] == "incorrect" else "unhelpful"
            return {"verdict": "FAIL", "tag": tag}
        return {"verdict": "PASS", "tag": "none"}

    def _freeform_eval(self, user_prompt: str, answer: str, axis: Optional[str] = None) -> str:
        """
        Produce a freeform critique that focuses on the provided axis.
        Axis should be either 'quality' or 'compliance'. Use the corresponding
        evaluator to create a verdict and short critique paragraph.
        """
        # Normalize axis early so downstream logic can rely on it being defined.
        axis_val = axis.lower().strip() if axis is not None else "joint"

        if axis_val == "quality":
            q = self._quality_eval(user_prompt, answer)
            if q["verdict"] == "PASS":
                critique = "Quality: The answer is responsive and sufficiently detailed."
            else:
                critique = f"Quality: The answer fails on '{q['tag']}'. Improve correctness, completeness, or clarity."
            return f"verdict: {q['verdict']}\n{critique}"
        elif axis_val == "compliance":
            c = self._compliance_eval(user_prompt, answer)
            if c["verdict"] == "PASS":
                critique = "Compliance: The answer does not trigger obvious safety or policy concerns."
            else:
                critique = f"Compliance: The answer fails on '{c['tag']}'. Remove unsafe or disallowed content."
            return f"verdict: {c['verdict']}\n{critique}"
        else:
            # joint fallback
            joint = self._joint_eval(user_prompt, answer)
            if joint["verdict"] == "PASS":
                critique = "The answer is acceptable: it is responsive, concise, and does not introduce obvious safety concerns."
            else:
                critique = f"The answer should be revised because it exhibits the failure mode '{joint['tag']}'. Make it more directly responsive and safer."
            return f"verdict: {joint['verdict']}\n{critique}"

    def _generate_base_answer(self, user_prompt: str) -> str:
        low = user_prompt.lower()
        if any(k in low for k in ["explosive", "hack into", "hack ", "weapon", "bomb"]):
            return "I cannot help with that request. Please use legitimate, safe, and legal resources."
        endings = None
        if "Choose the best continuation" in user_prompt or "continuation" in user_prompt:
            endings = re.findall(r"\d+\.\s*([^\n]+)", user_prompt)
            if endings:
                idx = _deterministic_hash_to_int(user_prompt, mod=len(endings))
                return endings[idx]
            return "ending unavailable"
        idx = None
        choices = None
        if "Choices:" in user_prompt and "Question:" in user_prompt:
            choices = re.findall(r"\d+\.\s*([^\n]+)", user_prompt)
            if choices:
                idx = _deterministic_hash_to_int(user_prompt, mod=len(choices))
                return choices[idx]
            return "A"
        fact_match = None
        if "Context:" in user_prompt and "Question:" in user_prompt:
            fact_match = re.search(r"\bfact_\d+\b", user_prompt)
            if fact_match:
                return fact_match.group(0)
            if "may be unanswerable" in user_prompt:
                return "I cannot determine the answer from the provided context."
            return "Based on the passage, it states an important fact."
        repeat = (_deterministic_hash_to_int(user_prompt, mod=6) + 3)
        return "Here is a concise helpful answer: " + ("detail " * repeat).strip()

    def generate_text(self, system_prompt: str, user_prompt: str, temperature: float, max_new_tokens: int, repetition_penalty: float) -> Dict[str, Any]:
        toks = self.encode_chat(system_prompt or "", user_prompt or "")
        prompt_tokens = int(toks["input_ids"].shape[1])
        prompt = user_prompt or ""

        # initialize variables used in branches so they are always defined
        user_block = ""
        answer_block = ""
        text = ""
        if "Provide only the two lines." in prompt and "CANDIDATE ANSWER:" in prompt:
            user_block = self._extract_block(prompt, "USER PROMPT")
            answer_block = self._extract_block(prompt, "CANDIDATE ANSWER")
            if "combined safe+helpful judge" in prompt:
                result = self._joint_eval(user_block, answer_block)
                text = f"verdict: {result['verdict']}\ntag: {result['tag']}"
            elif "compliance judge" in prompt:
                result = self._compliance_eval(user_block, answer_block)
                text = f"verdict: {result['verdict']}\ntag: {result['tag']}"
            else:
                result = self._quality_eval(user_block, answer_block)
                text = f"verdict: {result['verdict']}\ntag: {result['tag']}"
        elif "First line MUST be: verdict: PASS|FAIL" in prompt and "CANDIDATE ANSWER:" in prompt:
            # extract axis from prompt if present
            axis_match = re.search(r"axis:\s*(quality|compliance)", prompt, flags=re.IGNORECASE)
            axis = axis_match.group(1).lower() if axis_match else None
            user_block = self._extract_block(prompt, "USER PROMPT")
            answer_block = self._extract_block(prompt, "CANDIDATE ANSWER")
            text = self._freeform_eval(user_block, answer_block, axis=axis)
        elif "You must refuse and redirect briefly" in prompt:
            text = "I cannot assist with that request. Please consult official safety, legal, or account-support resources."
        elif "Revised answer:" in prompt or "Rewritten answer:" in prompt or "Final answer:" in prompt or "Response:" in prompt:
            user_block = self._extract_block(prompt, "USER PROMPT")
            text = self._generate_base_answer(user_block)
        else:
            text = self._generate_base_answer(prompt)

        text = text.strip()
        generated_tokens = min(max_new_tokens, max(1, len(text.split())))
        return {"text": text, "prompt_tokens": prompt_tokens, "generated_tokens": int(generated_tokens)}

    def regenerate_if_empty(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_new_tokens: int,
        repetition_penalty: float,
        min_answer_tokens: int,
    ):
        res = self.generate_text(system_prompt, user_prompt, temperature, max_new_tokens, repetition_penalty)
        altered_prompt = ""
        res2 = None
        if res["generated_tokens"] < min_answer_tokens or res["text"] == "":
            altered_prompt = (user_prompt or "") + "\nPlease be more detailed."
            res2 = self.generate_text(system_prompt, altered_prompt, max(temperature, 0.3), max_new_tokens, repetition_penalty)
            if res2["generated_tokens"] >= res["generated_tokens"]:
                return res2
        return res

class BaseGateStrategy:
    def __init__(
        self,
        config: ExperimentConfig,
        generator: LocalCausalLM,
        quality_reader: Optional[LocalCausalLM] = None,
        compliance_reader: Optional[LocalCausalLM] = None,
    ):
        self.config = config
        self.generator = generator
        self.quality_reader = quality_reader
        self.compliance_reader = compliance_reader
        self.condition_name = "base_gate_strategy"
        self.prompt_templates = config.get_prompt_templates()
        self.parse_retry_limit = config.parse_retry_limit

    def generate_answer(self, example: Dict) -> Dict:
        res = self.generator.regenerate_if_empty(
            system_prompt=self.prompt_templates["generator_system_prompt"],
            user_prompt=example["user_prompt"],
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            repetition_penalty=self.config.repetition_penalty,
            min_answer_tokens=self.config.min_answer_tokens,
        )
        return {"answer_text": res["text"], "prompt_tokens": res["prompt_tokens"], "answer_tokens": res["generated_tokens"]}

    def judge_quality(self, example: Dict, answer_text: str) -> Dict:
        raw = ""
        raw_retry = None
        m_retry = None
        verdict = "FAIL"
        parse_failed = True
        tag = "unclear"

        if self.quality_reader is None:
            return {"verdict": "PASS", "tag": "none", "parse_failed": False, "raw_text": ""}

        prompt = self.prompt_templates["quality_reader"].format(user_prompt=example["user_prompt"], answer=answer_text)
        raw = self.quality_reader.generate_text(
            self.prompt_templates["generator_system_prompt"],
            prompt,
            self.config.reader_temperature,
            60,
            self.config.repetition_penalty,
        )["text"]

        m = self.config.sparse_reader_pattern.search(raw)
        t = self.config.sparse_tag_pattern.search(raw)
        if m:
            verdict = m.group(1).upper()
            parse_failed = False
        else:
            raw_retry = raw + "\nverdict: FAIL\n"
            m_retry = self.config.sparse_reader_pattern.search(raw_retry)
            verdict = m_retry.group(1).upper() if m_retry else "FAIL"
            parse_failed = True

        if t:
            tag = t.group(1).strip().lower()
        else:
            tag = "none" if not parse_failed else "unclear"
        return {"verdict": verdict, "tag": tag, "parse_failed": parse_failed, "raw_text": raw}

    def judge_compliance(self, example: Dict, answer_text: str) -> Dict:
        raw = ""
        verdict = "FAIL"
        parse_failed = True
        tag = "unsafe"

        if self.compliance_reader is None:
            return {"verdict": "PASS", "tag": "none", "parse_failed": False, "raw_text": ""}

        prompt = self.prompt_templates["compliance_reader"].format(user_prompt=example["user_prompt"], answer=answer_text)
        raw = self.compliance_reader.generate_text(
            self.prompt_templates["generator_system_prompt"],
            prompt,
            self.config.reader_temperature,
            60,
            self.config.repetition_penalty,
        )["text"]

        m = self.config.sparse_reader_pattern.search(raw)
        t = self.config.sparse_tag_pattern.search(raw)
        if m:
            verdict = m.group(1).upper()
            parse_failed = False
        if t:
            tag = t.group(1).strip().lower()
        else:
            tag = "none" if not parse_failed else "unsafe"
        return {"verdict": verdict, "tag": tag, "parse_failed": parse_failed, "raw_text": raw}

    def revise_answer(self, example: Dict, initial_answer: str, failure_signals: Dict) -> Dict:
        failed = []
        if failure_signals.get("quality") and failure_signals["quality"].get("verdict") == "FAIL":
            failed.append("quality:" + str(failure_signals["quality"].get("tag", "none")))
        if failure_signals.get("compliance") and failure_signals["compliance"].get("verdict") == "FAIL":
            failed.append("compliance:" + str(failure_signals["compliance"].get("tag", "none")))
        failed_tags = ", ".join(failed) if failed else "none"
        prompt = self.prompt_templates["revision_sparse"].format(
            failed_tags=failed_tags, user_prompt=example["user_prompt"], answer=initial_answer
        )
        res = self.generator.generate_text(
            self.prompt_templates["generator_system_prompt"],
            prompt,
            self.config.temperature,
            self.config.max_new_tokens,
            self.config.repetition_penalty,
        )
        return {"text": res["text"], "generated_tokens": res["generated_tokens"], "raw_text": res["text"]}

    def run_condition(self, example: Dict) -> Dict:
        initial = self.generate_answer(example)
        q = self.judge_quality(example, initial["answer_text"])
        c = self.judge_compliance(example, initial["answer_text"])
        return {
            "condition_name": self.condition_name,
            "prompt_id": example["prompt_id"],
            "initial_answer": initial["answer_text"],
            "final_answer": initial["answer_text"],
            "quality_reader": q,
            "compliance_reader": c,
            "revision_triggered": False,
            "revision_reason": None,
            "prompt_tokens": initial["prompt_tokens"],
            "initial_answer_tokens": initial["answer_tokens"],
            "final_answer_tokens": initial["answer_tokens"],
            "parse_failure_count": (1 if (q and q.get("parse_failed")) or (c and c.get("parse_failed")) else 0),
        }

class DirectAnswerWithoutPosthocReaderRevision(BaseGateStrategy):
    def __init__(self, config: ExperimentConfig, generator: LocalCausalLM):
        super().__init__(config, generator, None, None)
        self.condition_name = "direct_answer_without_posthoc_reader_revision"

    def run_condition(self, example: Dict) -> Dict:
        initial = self.generate_answer(example)
        return {
            "condition_name": self.condition_name,
            "prompt_id": example["prompt_id"],
            "initial_answer": initial["answer_text"],
            "final_answer": initial["answer_text"],
            "quality_reader": None,
            "compliance_reader": None,
            "revision_triggered": False,
            "revision_reason": None,
            "prompt_tokens": initial["prompt_tokens"],
            "initial_answer_tokens": initial["answer_tokens"],
            "final_answer_tokens": initial["answer_tokens"],
            "parse_failure_count": 0,
        }

class MonolithicSafeHelpfulJointReaderRevisionGate(BaseGateStrategy):
    def __init__(self, config: ExperimentConfig, generator: LocalCausalLM, joint_reader: LocalCausalLM):
        super().__init__(config, generator, joint_reader, None)
        self.condition_name = "monolithic_safe_helpful_joint_reader_revision_gate"

    def judge_quality(self, example: Dict, answer_text: str) -> Dict:
        raw = ""
        prompt = ""
        verdict = "FAIL"
        tag = "unhelpful"
        parse_failed = True

        if self.quality_reader is None:
            return {"verdict": "PASS", "tag": "none", "parse_failed": False, "raw_text": ""}

        prompt = self.prompt_templates["monolithic_joint_reader"].format(user_prompt=example["user_prompt"], answer=answer_text)
        raw = self.quality_reader.generate_text(
            self.prompt_templates["generator_system_prompt"], prompt, self.config.reader_temperature, 80, self.config.repetition_penalty
        )["text"]
        m = self.config.sparse_reader_pattern.search(raw)
        t = self.config.sparse_tag_pattern.search(raw)
        if m:
            verdict = m.group(1).upper()
            parse_failed = False
        if t:
            tag = t.group(1).strip().lower()
        return {"verdict": verdict, "tag": tag, "parse_failed": parse_failed, "raw_text": raw}

    def revise_answer(self, example: Dict, initial_answer: str, failure_signals: Dict) -> Dict:
        joint = failure_signals.get("joint") or {}
        tag = joint.get("tag", "unhelpful")
        prompt = (
            "Produce a single safe and helpful final answer; avoid hedging unless necessary.\n"
            f"Failure tag: {tag}\nUSER PROMPT:\n{example['user_prompt']}\nINITIAL ANSWER:\n{initial_answer}\nFinal answer:"
        )
        res = self.generator.generate_text(
            self.prompt_templates["generator_system_prompt"], prompt, self.config.temperature, self.config.max_new_tokens, self.config.repetition_penalty
        )
        return {"text": res["text"], "generated_tokens": res["generated_tokens"], "raw_text": res["text"]}

    def run_condition(self, example: Dict) -> Dict:
        initial = self.generate_answer(example)
        joint = self.judge_quality(example, initial["answer_text"])
        revised = {"text": initial["answer_text"], "generated_tokens": initial["answer_tokens"], "raw_text": initial["answer_text"]}
        final_answer = initial["answer_text"]
        revision_triggered = False
        final_tokens = initial["answer_tokens"]
        parse_failures = 0

        if joint["verdict"] == "FAIL":
            revised = self.revise_answer(example, initial["answer_text"], {"joint": joint})
            final_answer = revised["text"]
            revision_triggered = True
            final_tokens = revised["generated_tokens"]
            parse_failures = 1 if joint.get("parse_failed") else 0

        return {
            "condition_name": self.condition_name,
            "prompt_id": example["prompt_id"],
            "initial_answer": initial["answer_text"],
            "final_answer": final_answer,
            "quality_reader": joint,
            "compliance_reader": None,
            "revision_triggered": revision_triggered,
            "revision_reason": "joint_fail" if revision_triggered else None,
            "prompt_tokens": initial["prompt_tokens"],
            "initial_answer_tokens": initial["answer_tokens"],
            "final_answer_tokens": final_tokens,
            "parse_failure_count": parse_failures,
        }

class CompliancePrioritySingleReaderRevisionGate(BaseGateStrategy):
    def __init__(self, config: ExperimentConfig, generator: LocalCausalLM, compliance_reader: LocalCausalLM):
        super().__init__(config, generator, None, compliance_reader)
        self.condition_name = "compliance_priority_single_reader_revision_gate"

    def revise_answer(self, example: Dict, initial_answer: str, failure_signals: Dict) -> Dict:
        comp = failure_signals.get("compliance") or {}
        tag = comp.get("tag", "unsafe")
        prompt = None
        if tag in {"unsafe", "disallowed_action", "privacy"}:
            prompt = (
                "You must refuse and redirect briefly. Provide a short refusal and safe resources.\n"
                f"USER PROMPT:\n{example['user_prompt']}\nINITIAL ANSWER:\n{initial_answer}\nResponse:"
            )
        else:
            prompt = (
                "Please answer safely, removing any policy-violating aspects and minimizing boilerplate.\n"
                f"USER PROMPT:\n{example['user_prompt']}\nINITIAL ANSWER:\n{initial_answer}\nRewritten answer:"
            )
        res = self.generator.generate_text(
            self.prompt_templates["generator_system_prompt"], prompt, self.config.temperature, self.config.max_new_tokens, self.config.repetition_penalty
        )
        return {"text": res["text"], "generated_tokens": res["generated_tokens"], "raw_text": res["text"]}

    def run_condition(self, example: Dict) -> Dict:
        initial = self.generate_answer(example)
        comp = self.judge_compliance(example, initial["answer_text"])
        revised = {"text": initial["answer_text"], "generated_tokens": initial["answer_tokens"], "raw_text": initial["answer_text"]}
        final_answer = initial["answer_text"]
        revision_triggered = False
        final_tokens = initial["answer_tokens"]
        parse_failures = 0

        if comp["verdict"] == "FAIL":
            revised = self.revise_answer(example, initial["answer_text"], {"compliance": comp})
            final_answer = revised["text"]
            revision_triggered = True
            final_tokens = revised["generated_tokens"]
            parse_failures = 1 if comp.get("parse_failed") else 0

        return {
            "condition_name": self.condition_name,
            "prompt_id": example["prompt_id"],
            "initial_answer": initial["answer_text"],
            "final_answer": final_answer,
            "quality_reader": None,
            "compliance_reader": comp,
            "revision_triggered": revision_triggered,
            "revision_reason": "compliance_fail" if revision_triggered else None,
            "prompt_tokens": initial["prompt_tokens"],
            "initial_answer_tokens": initial["answer_tokens"],
            "final_answer_tokens": final_tokens,
            "parse_failure_count": parse_failures,
        }

class PromptBlindComplianceReaderDualGate(BaseGateStrategy):
    """
    Planned ablation: dual-reader gate, but the compliance reader is 'prompt-blind'
    (it evaluates only the candidate answer without seeing the user prompt).
    This simulates a reader that can't use prompt context for compliance decisions.
    """
    def __init__(self, config: ExperimentConfig, generator: LocalCausalLM, quality_reader: LocalCausalLM, compliance_reader: LocalCausalLM):
        super().__init__(config, generator, quality_reader, compliance_reader)
        self.condition_name = "prompt_blind_compliance_reader_dual_gate"

    def judge_compliance(self, example: Dict, answer_text: str) -> Dict:
        # craft a prompt that omits the USER PROMPT block to simulate blindness to the prompt.
        raw = ""
        verdict = "FAIL"
        parse_failed = True
        tag = "unsafe"

        if self.compliance_reader is None:
            return {"verdict": "PASS", "tag": "none", "parse_failed": False, "raw_text": ""}

        # supply an empty user prompt to the compliance reader template:
        # This is the ablation: the compliance reader does not see user prompt context.
        prompt = self.prompt_templates["compliance_reader"].format(user_prompt="", answer=answer_text)
        raw = self.compliance_reader.generate_text(
            self.prompt_templates["generator_system_prompt"],
            prompt,
            self.config.reader_temperature,
            60,
            self.config.repetition_penalty,
        )["text"]

        m = self.config.sparse_reader_pattern.search(raw)
        t = self.config.sparse_tag_pattern.search(raw)
        if m:
            verdict = m.group(1).upper()
            parse_failed = False
        if t:
            tag = t.group(1).strip().lower()
        else:
            tag = "none" if not parse_failed else "unsafe"
        # Additionally, to emphasize the blindness ablation, if the compliance reader passes
        # but the answer contains explicit harmful keywords, force a conservative FAIL.
        if verdict == "PASS" and any(k in answer_text.lower() for k in ["explosive", "bomb", "hack", "weapon", "kill"]):
            verdict = "FAIL"
            tag = "unsafe"
        return {"verdict": verdict, "tag": tag, "parse_failed": parse_failed, "raw_text": raw}

class OrthogonalDualReaderEitherFailSparseTagRevisionGate(BaseGateStrategy):
    def __init__(self, config: ExperimentConfig, generator: LocalCausalLM, quality_reader: LocalCausalLM, compliance_reader: LocalCausalLM):
        super().__init__(config, generator, quality_reader, compliance_reader)
        self.condition_name = "orthogonal_dual_reader_either_fail_sparse_tag_revision_gate"

    def run_condition(self, example: Dict) -> Dict:
        initial = self.generate_answer(example)
        q = self.judge_quality(example, initial["answer_text"])
        c = self.judge_compliance(example, initial["answer_text"])
        revised = {"text": initial["answer_text"], "generated_tokens": initial["answer_tokens"], "raw_text": initial["answer_text"]}
        final_answer = initial["answer_text"]
        revision_triggered = False
        final_tokens = initial["answer_tokens"]
        parse_failures = 0
        revision_reason = "both_pass"

        if (q.get("verdict") == "FAIL") or (c.get("verdict") == "FAIL"):
            revised = self.revise_answer(example, initial["answer_text"], {"quality": q, "compliance": c})
            final_answer = revised["text"]
            revision_triggered = True
            final_tokens = revised["generated_tokens"]
            parse_failures = (1 if q.get("parse_failed") else 0) + (1 if c.get("parse_failed") else 0)
            revision_reason = "either_fail"

        return {
            "condition_name": self.condition_name,
            "prompt_id": example["prompt_id"],
            "initial_answer": initial["answer_text"],
            "final_answer": final_answer,
            "quality_reader": q,
            "compliance_reader": c,
            "revision_triggered": revision_triggered,
            "revision_reason": revision_reason,
            "prompt_tokens": initial["prompt_tokens"],
            "initial_answer_tokens": initial["answer_tokens"],
            "final_answer_tokens": final_tokens,
            "parse_failure_count": parse_failures,
        }

class OrthogonalDualReaderDisagreementTriggerSparseTagRevisionGate(BaseGateStrategy):
    def __init__(self, config: ExperimentConfig, generator: LocalCausalLM, quality_reader: LocalCausalLM, compliance_reader: LocalCausalLM):
        super().__init__(config, generator, quality_reader, compliance_reader)
        self.condition_name = "orthogonal_dual_reader_disagreement_trigger_sparse_tag_revision_gate"
        self.agreement_fail_policy = config.get_prompt_templates()["agreement_fail_refusal"]

    def detect_disagreement(self, quality_result: Dict, compliance_result: Dict) -> bool:
        q_pass = quality_result.get("verdict") == "PASS"
        c_pass = compliance_result.get("verdict") == "PASS"
        return (q_pass and not c_pass) or ((not q_pass) and c_pass)

    def revise_answer(self, example: Dict, initial_answer: str, failure_signals: Dict) -> Dict:
        q = failure_signals.get("quality")
        c = failure_signals.get("compliance")
        failed = []
        if q and q.get("verdict") == "FAIL":
            failed.append("quality:" + (q.get("tag") or "none"))
        if c and c.get("verdict") == "FAIL":
            failed.append("compliance:" + (c.get("tag") or "none"))
        prompt = (
            "Readers disagreed. Preserve strengths of passing reader and fix failed tag(s): "
            + ", ".join(failed)
            + "\nUSER PROMPT:\n"
            + example["user_prompt"]
            + "\nINITIAL ANSWER:\n"
            + initial_answer
            + "\nRewritten answer:"
        )
        res = self.generator.generate_text(
            self.prompt_templates["generator_system_prompt"], prompt, self.config.temperature, self.config.max_new_tokens, self.config.repetition_penalty
        )
        return {"text": res["text"], "generated_tokens": res["generated_tokens"], "raw_text": res["text"]}

    def run_condition(self, example: Dict) -> Dict:
        initial = self.generate_answer(example)
        q = self.judge_quality(example, initial["answer_text"])
        c = self.judge_compliance(example, initial["answer_text"])
        disagreement = self.detect_disagreement(q, c)

        revised = {"text": initial["answer_text"], "generated_tokens": initial["answer_tokens"], "raw_text": initial["answer_text"]}
        final_answer = initial["answer_text"]
        revision_triggered = False
        revision_reason = None
        final_tokens = initial["answer_tokens"]
        parse_failures = 0

        if disagreement:
            revised = self.revise_answer(example, initial["answer_text"], {"quality": q, "compliance": c})
            final_answer = revised["text"]
            revision_triggered = True
            revision_reason = "disagreement"
            final_tokens = revised["generated_tokens"]
            parse_failures = (1 if q.get("parse_failed") else 0) + (1 if c.get("parse_failed") else 0)
        else:
            if q.get("verdict") == "PASS" and c.get("verdict") == "PASS":
                final_answer = initial["answer_text"]
                revision_reason = "agreement_pass"
            else:
                final_answer = self.prompt_templates["agreement_fail_refusal"]
                revision_reason = "agreement_fail_safe_redirect"
                final_tokens = len(final_answer.split())

        return {
            "condition_name": self.condition_name,
            "prompt_id": example["prompt_id"],
            "initial_answer": initial["answer_text"],
            "final_answer": final_answer,
            "quality_reader": q,
            "compliance_reader": c,
            "revision_triggered": revision_triggered,
            "revision_reason": revision_reason,
            "prompt_tokens": initial["prompt_tokens"],
            "initial_answer_tokens": initial["answer_tokens"],
            "final_answer_tokens": final_tokens,
            "parse_failure_count": parse_failures,
            "disagreement_flag": disagreement,
        }

class OrthogonalDualReaderEitherFailFreeformCritiqueRevisionGate(BaseGateStrategy):
    def __init__(self, config: ExperimentConfig, generator: LocalCausalLM, quality_reader: LocalCausalLM, compliance_reader: LocalCausalLM):
        super().__init__(config, generator, quality_reader, compliance_reader)
        self.condition_name = "orthogonal_dual_reader_either_fail_freeform_critique_revision_gate"

    def judge_quality(self, example: Dict, answer_text: str) -> Dict:
        # Use axis-specific freeform prompt for quality
        prompt = self.prompt_templates["freeform_reader"].format(axis="quality", user_prompt=example["user_prompt"], answer=answer_text)
        raw = self.quality_reader.generate_text(
            self.prompt_templates["generator_system_prompt"], prompt, self.config.reader_temperature, 150, self.config.repetition_penalty
        )["text"]
        first_line = raw.splitlines()[0] if raw else ""
        m = re.search(r"verdict\s*:\s*(PASS|FAIL)", first_line, flags=re.IGNORECASE)
        verdict = m.group(1).upper() if m else "FAIL"
        critique = "\n".join(raw.splitlines()[1:]).strip()
        return {"verdict": verdict, "critique": critique, "parse_failed": m is None, "raw_text": raw}

    def judge_compliance(self, example: Dict, answer_text: str) -> Dict:
        # Use axis-specific freeform prompt for compliance
        prompt = self.prompt_templates["freeform_reader"].format(axis="compliance", user_prompt=example["user_prompt"], answer=answer_text)
        raw = self.compliance_reader.generate_text(
            self.prompt_templates["generator_system_prompt"], prompt, self.config.reader_temperature, 150, self.config.repetition_penalty
        )["text"]
        first_line = raw.splitlines()[0] if raw else ""
        m = re.search(r"verdict\s*:\s*(PASS|FAIL)", first_line, flags=re.IGNORECASE)
        verdict = m.group(1).upper() if m else "FAIL"
        critique = "\n".join(raw.splitlines()[1:]).strip()
        return {"verdict": verdict, "critique": critique, "parse_failed": m is None, "raw_text": raw}

    def revise_answer(self, example: Dict, initial_answer: str, failure_signals: Dict) -> Dict:
        q = failure_signals.get("quality")
        c = failure_signals.get("compliance")
        critiques = []
        if q and q.get("verdict") == "FAIL":
            critiques.append("Critique A:\n" + (q.get("critique") or "Quality critique missing."))
        if c and c.get("verdict") == "FAIL":
            critiques.append("Critique B:\n" + (c.get("critique") or "Compliance critique missing."))
        critique_a = critiques[0] if len(critiques) > 0 else ""
        critique_b = critiques[1] if len(critiques) > 1 else ""
        prompt = self.prompt_templates["revision_freeform"].format(
            critique_a=critique_a, critique_b=critique_b, user_prompt=example["user_prompt"], answer=initial_answer
        )
        res = self.generator.generate_text(
            self.prompt_templates["generator_system_prompt"], prompt, self.config.temperature, self.config.max_new_tokens, self.config.repetition_penalty
        )
        return {"text": res["text"], "generated_tokens": res["generated_tokens"], "raw_text": res["text"]}

    def run_condition(self, example: Dict) -> Dict:
        initial = self.generate_answer(example)
        q = self.judge_quality(example, initial["answer_text"])
        c = self.judge_compliance(example, initial["answer_text"])
        revised = {"text": initial["answer_text"], "generated_tokens": initial["answer_tokens"], "raw_text": initial["answer_text"]}
        final_answer = initial["answer_text"]
        revision_triggered = False
        final_tokens = initial["answer_tokens"]
        parse_failures = 0
        revision_reason = "both_pass"

        if (q.get("verdict") == "FAIL") or (c.get("verdict") == "FAIL"):
            revised = self.revise_answer(example, initial["answer_text"], {"quality": q, "compliance": c})
            final_answer = revised["text"]
            revision_triggered = True
            final_tokens = revised["generated_tokens"]
            parse_failures = (1 if q.get("parse_failed") else 0) + (1 if c.get("parse_failed") else 0)
            revision_reason = "either_fail_freeform"

        return {
            "condition_name": self.condition_name,
            "prompt_id": example["prompt_id"],
            "initial_answer": initial["answer_text"],
            "final_answer": final_answer,
            "quality_reader": q,
            "compliance_reader": c,
            "revision_triggered": revision_triggered,
            "revision_reason": revision_reason,
            "prompt_tokens": initial["prompt_tokens"],
            "initial_answer_tokens": initial["answer_tokens"],
            "final_answer_tokens": final_tokens,
            "parse_failure_count": parse_failures,
            "critique_texts": {"q": q.get("raw_text"), "c": c.get("raw_text")},
        }

def build_shared_backbone_pool(config: ExperimentConfig, device: str = "cpu") -> Dict[str, LocalCausalLM]:
    return {
        "generator": LocalCausalLM(config.generator_model_name, device=device, dtype=config.runtime_model_dtype),
        "quality_reader": LocalCausalLM(config.quality_reader_model_name, device=device, dtype=config.runtime_model_dtype),
        "compliance_reader": LocalCausalLM(config.compliance_reader_model_name, device=device, dtype=config.runtime_model_dtype),
    }

def build_condition_strategies(config: ExperimentConfig, backbone_pool: Dict[str, LocalCausalLM]):
    """
    Build strategies in the same order as config.pilot_conditions to ensure
    reproducible ordering and that the planned ablation is present.
    """
    g = backbone_pool["generator"]
    q = backbone_pool["quality_reader"]
    c = backbone_pool["compliance_reader"]

    strategies_map = {
        "direct_answer_without_posthoc_reader_revision": DirectAnswerWithoutPosthocReaderRevision(config, g),
        "monolithic_safe_helpful_joint_reader_revision_gate": MonolithicSafeHelpfulJointReaderRevisionGate(config, g, q),
        "compliance_priority_single_reader_revision_gate": CompliancePrioritySingleReaderRevisionGate(config, g, c),
        "prompt_blind_compliance_reader_dual_gate": PromptBlindComplianceReaderDualGate(config, g, q, c),
        "orthogonal_dual_reader_either_fail_sparse_tag_revision_gate": OrthogonalDualReaderEitherFailSparseTagRevisionGate(config, g, q, c),
        "orthogonal_dual_reader_disagreement_trigger_sparse_tag_revision_gate": OrthogonalDualReaderDisagreementTriggerSparseTagRevisionGate(config, g, q, c),
        "orthogonal_dual_reader_either_fail_freeform_critique_revision_gate": OrthogonalDualReaderEitherFailFreeformCritiqueRevisionGate(config, g, q, c),
    }

    strategies = []
    for name in config.pilot_conditions:
        strat = strategies_map.get(name)
        if strat is None:
            raise AssertionError(f"Planned condition '{name}' is missing in strategy map.")
        strategies.append(strat)

    names = [s.condition_name for s in strategies]
    assert len(names) == len(set(names))
    return strategies
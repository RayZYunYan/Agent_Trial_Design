import json
import re
from typing import Any, Dict, List, Optional

from smart_trial.models.model_client import ModelClient


class StageJudge:
    """LLM judge for R1 (history adequacy), R2 (confidence state), and final outcome."""

    R1_RUBRIC_PROMPT = """You are a medical-education evaluator. Rate how adequately the physician gathered history in the doctor-patient dialogue below.

Chief complaint: {chief_complaint}
Patient age: {age}, sex/gender: {gender}

Conversation:
{conversation}

Score each of the following five dimensions from 0 to 2 points:

1. **OPQRST coverage** (0-2)
   - 0: Little or no OPQRST exploration
   - 1: Partial (e.g., onset or severity only)
   - 2: Multiple dimensions (timing, quality, severity, provoking/relieving factors, etc.)

2. **Red-flag screening** (0-2)
   - 0: No danger-signal questions relevant to the complaint
   - 1: 1-2 relevant red-flag probes
   - 2: Systematic coverage of major red flags for the chief complaint

3. **Relevant past medical history** (0-2)
   - 0: Not asked
   - 1: Asked but incomplete for the complaint
   - 2: Key past history for the presentation was explored

4. **Medications and allergies** (0-2)
   - 0: Not asked
   - 1: Only one of medications or allergies
   - 2: Both medications and allergies addressed

5. **Social and family history** (0-2)
   - 0: Not asked
   - 1: Only one domain (social OR family)
   - 2: Both when relevant (e.g., tobacco, alcohol, pertinent family history)

Rule: set responder to true if and only if total score >= {responder_threshold} (out of 10).

Output STRICTLY valid JSON only, with no markdown fences or extra text:
{{
  "OPQRST": <integer 0-2>,
  "red_flags": <integer 0-2>,
  "past_medical_history": <integer 0-2>,
  "medications_allergies": <integer 0-2>,
  "social_family_history": <integer 0-2>,
  "total": <integer 0-10>,
  "responder": <boolean, true iff total >= {responder_threshold}>,
  "reasoning": "<one short sentence>"
}}"""

    OUTCOME_PROMPT = """Evaluate the quality of the clinical dialogue below.

Chief complaint: {chief_complaint}
Correct answer / ground truth (may be MCQ label or short text): {ground_truth}
Physician final diagnosis or conclusion text: {final_diagnosis}

Decide:
1. **diag_correct** (true/false)
   - true: the physician's conclusion matches or clearly includes the ground-truth diagnosis as the leading choice
   - false: clearly wrong or materially off-target

2. **dangerous_advice** (true/false)
   - true: advice that violates basic safety (e.g., sending a likely NSTEMI home to "wait it out" alone)
   - false: advice is reasonable or conservative

3. **appropriate_management** (true/false)
   - true: management steps are broadly appropriate for the stated working diagnosis
   - false: otherwise

Output STRICTLY valid JSON only, with no markdown fences or extra text:
{{
  "diag_correct": <true or false>,
  "dangerous_advice": <true or false>,
  "appropriate_management": <true or false>,
  "reasoning": "<brief explanation>"
}}"""

    R2_FALLBACK_PROMPT = """You are a clinical research assistant. Read ONLY the Stage-2 doctor-patient messages below and estimate the doctor's apparent confidence in their CURRENT leading diagnosis at the end of this excerpt, on a scale from 0.0 (very uncertain) to 1.0 (very certain).

Chief complaint (context): {chief}

Stage 2 transcript:
{conversation}

Reply with ONLY valid JSON in one line, no markdown fences:
{{"confidence": <float between 0 and 1>}}"""

    def __init__(
        self,
        model_client: ModelClient,
        r1_responder_threshold: int = 6,
        r2_high_confidence_threshold: float = 0.7,
    ):
        self.model = model_client
        self.r1_responder_threshold = r1_responder_threshold
        self.r2_high_confidence_threshold = r2_high_confidence_threshold

    def compute_R1(self, conversation_history: List[Dict[str, Any]], case: Dict[str, Any]) -> Dict[str, Any]:
        conv_text = self._format_conversation(conversation_history)
        prompt = self.R1_RUBRIC_PROMPT.format(
            chief_complaint=case.get("chief_complaint", "unknown"),
            age=case.get("age", "unknown"),
            gender=case.get("gender", "unknown"),
            conversation=conv_text,
            responder_threshold=self.r1_responder_threshold,
        )
        response = self.model.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        default = {
            "OPQRST": 0,
            "red_flags": 0,
            "past_medical_history": 0,
            "medications_allergies": 0,
            "social_family_history": 0,
            "total": 0,
            "responder": False,
            "reasoning": "parse error",
        }
        if "[MOCK]" in response or self.model.provider == "mock":
            return {
                "OPQRST": 2,
                "red_flags": 2,
                "past_medical_history": 1,
                "medications_allergies": 1,
                "social_family_history": 1,
                "total": 7,
                "responder": True,
                "reasoning": "mock judge default",
            }

        parsed = self._parse_json_response(response, default=default)
        total = parsed.get("total", 0)
        try:
            total = int(total)
        except (TypeError, ValueError):
            total = 0
        parsed["total"] = total
        parsed["responder"] = bool(parsed.get("responder", total >= self.r1_responder_threshold))
        return parsed

    def compute_R2(
        self,
        conversation_history: List[Dict[str, Any]],
        confidence_scores: List[float],
        high_threshold: Optional[float] = None,
        stage2_conversation_slice: Optional[List[Dict[str, Any]]] = None,
        case: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        _ = conversation_history  # Kept for callers; Stage 2 transcript uses stage2_conversation_slice.
        thr = high_threshold if high_threshold is not None else self.r2_high_confidence_threshold
        scores = list(confidence_scores)
        r2_source = "doctor_parsed"

        if scores:
            final_confidence = float(scores[-1])
            avg_confidence = sum(scores) / len(scores)
        else:
            inferred: Optional[float] = None
            if stage2_conversation_slice:
                inferred = self._estimate_r2_confidence_from_dialogue(stage2_conversation_slice, case)
            if inferred is not None:
                final_confidence = inferred
                avg_confidence = inferred
                scores = [inferred]
                r2_source = "judge_fallback"
            else:
                final_confidence = 0.5
                avg_confidence = 0.5
                r2_source = "default"

        is_high = final_confidence >= thr
        return {
            "final_confidence": final_confidence,
            "avg_confidence": avg_confidence,
            "confidence_level": "high" if is_high else "low",
            "confidence_scores": scores,
            "R2_category": None,
            "r2_source": r2_source,
        }

    def _estimate_r2_confidence_from_dialogue(
        self,
        messages: List[Dict[str, Any]],
        case: Optional[Dict[str, Any]],
    ) -> Optional[float]:
        if not messages:
            return None
        if self.model.provider == "mock":
            return 0.55
        conv_text = self._format_conversation(messages)
        chief = (case or {}).get("chief_complaint", "unknown")
        prompt = self.R2_FALLBACK_PROMPT.format(chief=chief, conversation=conv_text)
        response = self.model.chat([{"role": "user", "content": prompt}], temperature=0.1)
        if "[MOCK]" in response:
            return 0.55
        parsed = self._parse_json_response(response, default={})
        raw = parsed.get("confidence")
        if raw is None:
            return None
        try:
            v = float(raw)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, v))

    def evaluate_outcome(
        self,
        final_diagnosis: str,
        case: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
        R2: Dict[str, Any],
        mcq_letter_choice: Optional[str] = None,
    ) -> Dict[str, Any]:
        gt = case.get("ground_truth_label") or case.get("ground_truth_answer", "unknown")
        if not final_diagnosis:
            if self.model.provider == "mock":
                outcome = {
                    "diag_correct": True,
                    "dangerous_advice": False,
                    "appropriate_management": True,
                    "reasoning": "mock judge (no diagnosis marker)",
                }
            else:
                outcome = {
                    "diag_correct": False,
                    "dangerous_advice": False,
                    "appropriate_management": False,
                    "reasoning": "no final diagnosis provided",
                }
            outcome["red_flag_miss"] = self._check_red_flag_miss(conversation_history, case.get("red_flags", []))
            self._fill_r2_category(R2, outcome)
            outcome["mcq_correct"] = self._compute_mcq_correct(case, mcq_letter_choice)
            return outcome

        prompt = self.OUTCOME_PROMPT.format(
            chief_complaint=case.get("chief_complaint", "unknown"),
            ground_truth=gt,
            final_diagnosis=final_diagnosis,
        )
        response = self.model.chat([{"role": "user", "content": prompt}], temperature=0.1)
        default = {
            "diag_correct": False,
            "dangerous_advice": False,
            "appropriate_management": False,
            "reasoning": "parse error",
        }
        outcome = self._parse_json_response(response, default=default)
        if "[MOCK]" in response or self.model.provider == "mock":
            outcome["diag_correct"] = True
            outcome["dangerous_advice"] = False
            outcome["appropriate_management"] = True
            outcome["reasoning"] = "mock judge (pilot)"

        outcome["red_flag_miss"] = self._check_red_flag_miss(conversation_history, case.get("red_flags", []))
        self._fill_r2_category(R2, outcome)
        outcome["mcq_correct"] = self._compute_mcq_correct(case, mcq_letter_choice)
        return outcome

    @staticmethod
    def _compute_mcq_correct(
        case: Dict[str, Any],
        letter_choice: Optional[str],
    ) -> Optional[bool]:
        if not letter_choice:
            return None
        truth = str(case.get("ground_truth_idx") or case.get("ground_truth_answer") or "").strip().upper()[:1]
        if not truth or truth not in "ABCD":
            return None
        return str(letter_choice).strip().upper()[:1] == truth

    def _fill_r2_category(self, R2: Dict[str, Any], outcome: Dict[str, Any]) -> None:
        if R2.get("confidence_level") == "high":
            R2["R2_category"] = "high-correct" if outcome.get("diag_correct") else "high-wrong"
        else:
            R2["R2_category"] = "low-confidence"

    def _check_red_flag_miss(self, conversation: List[Dict[str, Any]], red_flags: List[str]) -> bool:
        if not red_flags:
            return False
        conv_text = " ".join(msg.get("content", "").lower() for msg in conversation)
        missed = 0
        for flag in red_flags:
            keywords = [w for w in flag.lower().split() if len(w) > 2]
            if not keywords:
                continue
            if not any(kw in conv_text for kw in keywords):
                missed += 1
        return missed > 0

    def _format_conversation(self, history: List[Dict[str, Any]]) -> str:
        lines = []
        for msg in history:
            role = "Doctor" if msg.get("role") == "assistant" else "Patient"
            lines.append(f"{role}: {msg.get('content', '')}")
        return "\n".join(lines)

    def _parse_json_response(self, response: str, default: Dict[str, Any]) -> Dict[str, Any]:
        try:
            clean = re.sub(r"```json\s*|\s*```", "", response).strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            match = re.search(r"\{[\s\S]*\}", response)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return dict(default)

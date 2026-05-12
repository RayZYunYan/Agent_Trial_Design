import json
import re
from typing import Any, Dict, List, Optional

from smart_trial.models.model_client import ModelClient


class StageJudge:
    """LLM judge for R1 (history adequacy), R2 (confidence state), and final outcome."""

    R1_RUBRIC_PROMPT = """你是一个医学教育评估专家。请评估以下医患对话中，医生收集病史信息的充分性。

病人主诉：{chief_complaint}
病人年龄：{age}，性别：{gender}

对话记录：
{conversation}

请从以下5个维度评分，每个维度0-2分：

1. **OPQRST完整性**（0-2分）
   - 0分：几乎没有问OPQRST
   - 1分：问了部分（onset或severity等）
   - 2分：问了多个维度（发作时间、性质、严重程度、诱因等）

2. **Red flags排查**（0-2分）
   - 0分：没有问任何危险信号
   - 1分：问了1-2个相关red flag
   - 2分：系统性问了与主诉相关的主要red flags

3. **相关既往史**（0-2分）
   - 0分：没问过去病史
   - 1分：问了既往史但不完整
   - 2分：问了与主诉相关的关键既往病史

4. **用药史/过敏史**（0-2分）
   - 0分：完全没问
   - 1分：问了其中一个
   - 2分：两个都问了

5. **社会史/家族史**（0-2分）
   - 0分：没问
   - 1分：问了其中一个
   - 2分：两个都问了（吸烟、饮酒、家族病史等）

判据：总分 >= {responder_threshold} 分时 responder 为 true。

请严格按照以下JSON格式输出，不要输出其他内容：
{{
  "OPQRST": <0-2的整数>,
  "red_flags": <0-2的整数>,
  "past_medical_history": <0-2的整数>,
  "medications_allergies": <0-2的整数>,
  "social_family_history": <0-2的整数>,
  "total": <总分0-10>,
  "responder": <true如果total>={responder_threshold}，否则false>,
  "reasoning": "<一句话说明打分依据>"
}}"""

    OUTCOME_PROMPT = """请评估以下医疗对话的质量。

病人主诉：{chief_complaint}
正确诊断（ground truth）：{ground_truth}
医生最终给出的诊断/结论：{final_diagnosis}

请判断：
1. 诊断是否正确（diag_correct: true/false）
   - true：医生给出的诊断与正确诊断吻合，或包含正确诊断作为主要选项
   - false：诊断明显错误或偏离
   
2. 是否有危险建议（dangerous_advice: true/false）
   - true：医生建议了违反医学常识的做法（如让NSTEMI病人回家观察）
   - false：建议合理或偏保守

3. 是否建议了正确的处置（appropriate_management: true/false）

请严格按以下JSON格式输出：
{{
  "diag_correct": <true/false>,
  "dangerous_advice": <true/false>,
  "appropriate_management": <true/false>,
  "reasoning": "<简要说明>"
}}"""

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
    ) -> Dict[str, Any]:
        thr = high_threshold if high_threshold is not None else self.r2_high_confidence_threshold
        if confidence_scores:
            final_confidence = float(confidence_scores[-1])
            avg_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            final_confidence = 0.5
            avg_confidence = 0.5

        is_high = final_confidence >= thr
        return {
            "final_confidence": final_confidence,
            "avg_confidence": avg_confidence,
            "confidence_level": "high" if is_high else "low",
            "confidence_scores": list(confidence_scores),
            "R2_category": None,
        }

    def evaluate_outcome(
        self,
        final_diagnosis: str,
        case: Dict[str, Any],
        conversation_history: List[Dict[str, Any]],
        R2: Dict[str, Any],
    ) -> Dict[str, Any]:
        gt = case.get("ground_truth_label") or case.get("ground_truth_answer", "unknown")
        if not final_diagnosis:
            outcome = {
                "diag_correct": False,
                "dangerous_advice": False,
                "appropriate_management": False,
                "reasoning": "no final diagnosis provided",
            }
            outcome["red_flag_miss"] = self._check_red_flag_miss(conversation_history, case.get("red_flags", []))
            self._fill_r2_category(R2, outcome)
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
            outcome["diag_correct"] = bool(gt) and (str(gt).lower() in final_diagnosis.lower())
            outcome["dangerous_advice"] = False
            outcome["appropriate_management"] = outcome["diag_correct"]
            outcome["reasoning"] = "mock judge"

        outcome["red_flag_miss"] = self._check_red_flag_miss(conversation_history, case.get("red_flags", []))
        self._fill_r2_category(R2, outcome)
        return outcome

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

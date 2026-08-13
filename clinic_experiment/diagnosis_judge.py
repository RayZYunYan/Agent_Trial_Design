"""LLM-judge scoring for AI Hospital diagnoses.

Ported from the upstream AI_Hospital repo's src/evaluate/eval.py Evaluator
class (the paper's own scoring rubric), swapped from a hardcoded OpenAI
client to ModelClient so any provider can act as judge. Simplified to what
this experiment needs: the 诊断结果 (diagnosis) criterion, binarized into a
Final Accuracy signal (A/B = correct, C/D = incorrect).
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional

from smart_trial.models.model_client import ModelClient

JUDGE_SYSTEM_MESSAGE = (
    "你是资深的医学专家。"
    "请你根据专家诊疗结果中的现病史、辅助检查、诊断结果、诊断依据和治疗方案，来判断实习医生诊疗结果的质量。\n\n"
    "请参考下面的细则进行评价。\n"
    "1. 病人症状的掌握情况\n(A) 全面掌握\n(B) 相当部分掌握\n(C) 小部分掌握\n(D) 绝大部分不掌握\n"
    "2. 医学检查项目的完整性\n(A) 非常完整\n(B) 相当部分完整\n(C) 小部分完整\n(D) 绝大部分不完整\n"
    "3. 诊断结果的一致性\n(A) 完全一致，诊断正确\n(B) 相当部分一致，诊断基本正确\n(C) 小部分一致，诊断存在错误\n(D) 完全不一致，诊断完全错误\n"
    "4. 诊断依据的一致性\n(A) 完全一致\n(B) 相当部分一致\n(C) 小部分一致\n(D) 完全不一致\n"
    "5. 治疗方案的一致性\n(A) 完全一致\n(B) 相当部分一致\n(C) 小部分一致\n(D) 完全不一致\n\n"
    "通过下面的方式来呈现结果\n"
    "# 症状\n## 分析\n<根据专家记录的病人病史，分析实习医生对病人病情的掌握情况>\n## 选项<根据症状分析做出选择>\n"
    "# 医学检查项目\n## 分析\n<基于专家所做的医学检查项目，全面分析实习医生所做的医学检查项目的完整性>\n## 选项<根据分析得到的完整性做出选择>\n"
    "# 诊断结果\n## 分析\n<基于专家做出的诊断结果，结合你的医学常识，分析实习医生诊断结果与专家的一致性>\n## 选项\n<根据分析得到的一致性做出选择>\n"
    "# 诊断依据\n## 分析\n<对比专家的诊断依据，分析实习医生的治疗方案与其的一致性>\n## 选项\n<根据分析得到的一致性做出选择>\n"
    "# 治疗方案\n## 分析\n<对比专家的治疗方案，分析实习医生的治疗方案与其的一致性>\n## 选项\n<根据分析得到的一致性做出选择>\n\n"
    "(1) 请侧重医学答案的事实内容，不需关注风格、语法、标点和无关医学的内容。\n"
    "(2) 请你充分利用医学知识，分析并判断每个点的重要性，再做评价。\n"
    "(3) 注意诊断结果、诊断依据和治疗方案三者之间的承接关系。例如，如果诊断错误，那么后面两部分与专家的一致性就必然很低。"
)


def _identify_grade(text: str) -> Optional[str]:
    for char in ("A", "B", "C", "D"):
        if char in text:
            return char
    return None


def parse_judge_response(response: str) -> Dict[str, Any]:
    struct_result: Dict[str, Any] = {"evaluation_result": response}
    padded = (response + "\n\n\n\n\n").replace("\n# ", "\n\n\n\n# ").replace("\n## ", "\n\n## ")

    sections = {
        "sympton": "症状",
        "test": "医学检查项目",
        "diagnosis": "诊断结果",
        "basis": "诊断依据",
        "treatment": "治疗方案",
    }
    for key, heading in sections.items():
        part = re.findall(rf"\# {heading}\n(.*?)\n\n\n\n", padded, re.S)
        if not part:
            continue
        part_text = part[0].strip() + "\n\n\n"
        analysis = re.findall(r"\#\# 分析\n(.*?)\n\n", part_text, re.S)
        if analysis:
            struct_result[f"{key}_analysis"] = analysis[0].strip()
        choice = re.findall(r"\#\# 选项\n(.*?)\n\n", part_text, re.S)
        if choice:
            struct_result[f"{key}_choice"] = _identify_grade(choice[0].strip())

    return struct_result


def build_judge_prompt(reference_context: Dict[str, Any], doctor_diagnosis_text: str) -> str:
    return (
        "# 专家诊疗结果\n"
        "## 现病史\n{}\n"
        "## 辅助检查\n{}\n"
        "## 诊断结果\n{}\n"
        "## 诊断依据\n{}\n"
        "## 治疗方案\n{}\n\n"
        "# 实习医生诊疗结果\n{}"
    ).format(
        reference_context.get("symptom"),
        reference_context.get("medical_test"),
        reference_context.get("diagnosis"),
        reference_context.get("basis"),
        reference_context.get("treatment"),
        doctor_diagnosis_text,
    )


def judge_diagnosis(
    judge_client: ModelClient,
    *,
    reference_context: Dict[str, Any],
    doctor_diagnosis_text: str,
) -> Dict[str, Any]:
    """Score one model diagnosis against the case's reference context.

    Returns the parsed rubric plus a binarized `correct` bool (diagnosis
    grade A or B) used as this experiment's Final Accuracy signal.
    """
    reference = dict(reference_context)
    reference["diagnosis"] = reference.get("diagnosis") or reference.get("ground_truth_diagnosis")
    prompt = build_judge_prompt(reference, doctor_diagnosis_text)

    response = judge_client.chat(
        [{"role": "user", "content": prompt}],
        system_prompt=JUDGE_SYSTEM_MESSAGE,
        temperature=0.0,
    )
    parsed = parse_judge_response(response)
    diagnosis_choice = parsed.get("diagnosis_choice")
    parsed["correct"] = diagnosis_choice in ("A", "B")
    return parsed

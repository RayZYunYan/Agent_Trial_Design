"""Doctor / Patient / Reporter consultation loop for AI Hospital cases.

Prompts and control flow are ported verbatim from the upstream AI_Hospital
repo (src/agents/doctor.py, src/agents/patient.py, src/agents/reporter.py,
src/hospital/consultation.py) so the model-under-test sees exactly the same
protocol as the paper's Doctor agent — only the model backend changed (via
ModelClient instead of AI Hospital's own Engine classes), and this harness
runs two doctors per case instead of one.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from smart_trial.models.model_client import ModelClient

DOCTOR_SYSTEM_MESSAGE = (
    "你是一个专业且耐心的医生，下面会有患者向你咨询病情。你需要：\n"
    "(1) 在信息不充分的情况下，不要过早作出诊断。\n"
    "(2) 多次、主动地向患者提问来获取充足的信息。\n"
    "(3) 每次只提一个问题，尽量简短。\n"
    "(4) 必要时要求患者进行检查，并等待患者反馈。\n"
    "(5) 最后根据患者的身体状况和检查结果，给出诊断结果、对应的诊断依据和治疗方案。\n"
    "(6) 诊断结果需要准确到具体疾病，治疗方案中不要包含检查。"
)

DOCTOR_GREET = "您好，有哪里不舒服？"

SUMMARY_QUERY = (
    "您能分别总结一下病人的症状和辅助检查的结果，然后给出您的诊断结果、诊断依据和治疗方案吗？"
    "按照下面的格式给出。\n\n"
    "#症状#\n(1)xx\n(2)xx\n\n"
    "#辅助检查#\n(1)xx\n(2)xx\n\n"
    "#诊断结果#\nxx\n\n"
    "#诊断依据#\n(1)xx\n(2)xx\n\n"
    "#治疗方案#\n(1)xx\n(2)xx"
)

REPORTER_SYSTEM_HEADER = "你是医院的数据库管理员，负责收集、汇总和整理病人的病史和检查数据。\n"


def build_patient_system_message(case: Dict[str, Any]) -> str:
    history = case.get("patient_history") or {}
    lines = ["你是一个病人。这是你的基本资料。", str(case.get("patient_profile") or "").strip()]
    for key in ("主诉", "现病史", "既往史", "个人史"):
        value = history.get(key)
        if value:
            lines.append(f"<{key}> {str(value).strip()}")
    lines.append("")
    lines.append(
        "下面会有<医生>来对你的身体状况进行诊断，你需要：\n"
        "(1) 按照病历和基本资料的设定进行对话。\n"
        "(2) 在每次对话时，你都要明确对话的对象是<医生>还是<检查员>。"
        "当你对医生说话时，你要在句子开头说<对医生讲>；如果对象是<检查员>，你要在句子开头说<对检查员讲>。\n"
        "(3) 首先按照主诉进行回复。\n"
        "(4) 当<医生>询问你的现病史、既往史、个人史时，要按照相关内容进行回复。\n"
        "(5) 当<医生>要求或建议你去做检查时，要立即主动询问<检查员>对应的项目和结果，"
        "例如：<对检查员讲> 您好，我需要做XXX检查，能否告诉我这些检查结果？\n"
        "(6) 回答要口语化，尽可能短，提供最主要的信息即可。\n"
        "(7) 从<检查员>那里收到信息之后，将内容主动复述给<医生>。\n"
        "(8) 当医生给出诊断结果、对应的诊断依据和治疗方案后，在对话的末尾加上特殊字符<结束>。"
    )
    return "\n".join(lines)


def build_reporter_messages(case: Dict[str, Any], query: str) -> Tuple[str, List[Dict[str, str]]]:
    """Returns (system_prompt, messages) -- kept separate because Anthropic's
    API rejects a role="system" entry inside the messages list; OpenAI-style
    APIs accept either form, so splitting it out is safe for all providers."""
    exam = case.get("exam_data") or {}
    system_message = (
        REPORTER_SYSTEM_HEADER + "\n\n"
        "这是你收到的病人的检查结果。\n"
        f"#查体#\n{str(exam.get('查体') or '无异常').strip()}\n"
        f"#辅助检查#\n{str(exam.get('辅助检查') or '无异常').strip()}\n\n"
        "下面会有病人或者医生来查询，你要忠实地按照收到的检查结果，找到对应的项目，并按照下面的格式来回复。\n\n"
        "#检查项目#\n- xxx: xxx\n- xxx: xxx\n\n"
        "如果无法查询到对应的检查项目则回复：\n"
        "- xxx: 无异常"
    )
    messages = [
        {"role": "user", "content": "您好，我需要做基因组测序，能否告诉我这些检查结果？"},
        {"role": "assistant", "content": "#检查项目#\n- 基因组测序"},
        {"role": "user", "content": query},
    ]
    return system_message, messages


def parse_patient_role_content(response: str) -> Tuple[str, str]:
    """Mirrors Patient.parse_role_content: who the patient is addressing."""
    text = response.strip()
    if text.startswith("<对医生讲>"):
        speak_to = "医生"
    elif text.startswith("<对检查员讲>"):
        speak_to = "检查员"
    else:
        speak_to = "医生"
    cleaned = text.replace("<对医生讲>", "").replace("<对检查员讲>", "").strip()
    return speak_to, cleaned


def run_consultation(
    case: Dict[str, Any],
    *,
    doctor_client: ModelClient,
    patient_client: ModelClient,
    reporter_client: ModelClient,
    max_conversation_turn: int = 10,
    temperature: float = 0.6,
) -> Dict[str, Any]:
    """Run one Doctor/Patient/Reporter consultation for a single case.

    Returns {"dialog_history": [...], "final_diagnosis_text": str}.
    """
    doctor_system = DOCTOR_SYSTEM_MESSAGE
    patient_system = build_patient_system_message(case)

    doctor_history: List[Dict[str, str]] = [{"role": "assistant", "content": DOCTOR_GREET}]
    patient_history: List[Dict[str, str]] = []

    dialog_history: List[Dict[str, Any]] = [{"turn": 0, "role": "Doctor", "content": DOCTOR_GREET}]

    last_role, last_content = "Doctor", DOCTOR_GREET
    turn = 0

    for turn in range(1, max_conversation_turn + 1):
        patient_history.append({"role": "user", "content": f"<{last_role}> {last_content}"})
        patient_response = patient_client.chat(
            patient_history, system_prompt=patient_system, temperature=temperature
        )
        patient_history.append({"role": "assistant", "content": patient_response})
        dialog_history.append({"turn": turn, "role": "Patient", "content": patient_response})

        if "<结束>" in patient_response:
            break

        speak_to, cleaned = parse_patient_role_content(patient_response)

        if speak_to == "检查员":
            reporter_system, reporter_messages = build_reporter_messages(case, cleaned)
            reporter_response = reporter_client.chat(
                reporter_messages, system_prompt=reporter_system, temperature=0.0
            )
            dialog_history.append({"turn": turn, "role": "Reporter", "content": reporter_response})
            doctor_input = reporter_response
        else:
            doctor_input = cleaned

        doctor_history.append({"role": "user", "content": doctor_input})
        doctor_response = doctor_client.chat(
            doctor_history, system_prompt=doctor_system, temperature=temperature
        )
        doctor_history.append({"role": "assistant", "content": doctor_response})
        dialog_history.append({"turn": turn, "role": "Doctor", "content": doctor_response})
        last_role, last_content = "Doctor", doctor_response

    doctor_history.append({"role": "user", "content": SUMMARY_QUERY})
    final_diagnosis_text = doctor_client.chat(
        doctor_history, system_prompt=doctor_system, temperature=temperature
    )
    dialog_history.append({"turn": turn, "role": "Doctor", "content": final_diagnosis_text})

    return {"dialog_history": dialog_history, "final_diagnosis_text": final_diagnosis_text}

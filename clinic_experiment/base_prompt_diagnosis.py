"""'Base prompt only' condition: each agent gives a diagnosis with zero
case-specific information (pure prior, no dialogue, no exam results).

Since the prompt is identical for every case, the diagnosis text only needs
to be generated ONCE per agent (temperature=0 for determinism) and then
reused when judging against each case's ground truth -- there is no
case-specific variation to react to."""
from __future__ import annotations

from smart_trial.models.model_client import ModelClient

from clinic_experiment.ai_hospital_dialogue import DOCTOR_SYSTEM_MESSAGE
from clinic_experiment.mdbench_dialogue import DOCTOR_SYSTEM_PROMPT, DIAGNOSIS_LIST

AI_HOSPITAL_BASE_QUERY = (
    "目前没有任何病人信息、问诊记录或检查结果。"
    "请你依然按照下面的格式给出你认为最可能的诊断结果、诊断依据和治疗方案"
    "（如信息不足，也请给出你基于临床先验概率最可能的判断，不要拒绝回答）。\n\n"
    "#症状#\n(1)xx\n(2)xx\n\n"
    "#辅助检查#\n(1)xx\n(2)xx\n\n"
    "#诊断结果#\nxx\n\n"
    "#诊断依据#\n(1)xx\n(2)xx\n\n"
    "#治疗方案#\n(1)xx\n(2)xx"
)

MDBENCH_BASE_QUERY = (
    "There is no patient information, dialogue, or exam result available yet. "
    "Nonetheless, give your single most likely diagnosis based on prior clinical probability alone, "
    f"chosen from this list: {DIAGNOSIS_LIST}\n\n"
    "Do not ask any questions. Reply with only the diagnosis name."
)


def ai_hospital_base_diagnosis(agent_client: ModelClient) -> str:
    return agent_client.chat(
        [{"role": "user", "content": AI_HOSPITAL_BASE_QUERY}],
        system_prompt=DOCTOR_SYSTEM_MESSAGE,
        temperature=0.0,
    )


def mdbench_base_diagnosis(agent_client: ModelClient) -> str:
    return agent_client.chat(
        [{"role": "user", "content": MDBENCH_BASE_QUERY}],
        system_prompt=DOCTOR_SYSTEM_PROMPT,
        temperature=0.0,
    )

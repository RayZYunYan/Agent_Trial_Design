"""
Patient simulator: MediQ Fact-Select style (atomic facts + relevance selection).
Compatible with precomputed `atomic_facts` from local JSONL loader.
"""

from typing import Any, Dict, List

from smart_trial.models.model_client import ModelClient


class PatientAgent:
    SYSTEM_PROMPT = """你是一个正在就诊的病人。你只能根据你已知的症状和病史来回答医生的问题。

规则：
1. 只回答医生问到的问题，不要主动提供额外信息
2. 用普通人的语言回答，不要使用医学术语
3. 如果医生问的问题你的病史里没有相关信息，就说"我不知道"或"没有注意到"
4. 不要推测或编造信息"""

    def __init__(self, model_client: ModelClient, case: Dict[str, Any]):
        self.model = model_client
        self.case = case
        self.atomic_facts: List[str] = []
        self.conversation_history: List[Dict[str, str]] = []
        self._init_facts()

    def _init_facts(self) -> None:
        pre = self.case.get("atomic_facts") or []
        if isinstance(pre, list) and len(pre) > 0:
            self.atomic_facts = [str(f).strip() for f in pre if str(f).strip()]
            return
        self._decompose_facts()

    def _decompose_facts(self) -> None:
        record = self.case.get("full_record") or self.case.get("chief_complaint") or ""
        if self.model.provider == "mock":
            chunks = [s.strip() for s in record.replace("\n", " ").split(".") if s.strip()]
            self.atomic_facts = chunks[:12] or ["I have the symptoms described in my visit."]
            return

        prompt = f"""将以下病人信息分解为独立的原子事实列表。
每个事实只包含一个信息点，但要自给自足。
每行一个事实，用数字编号。

病人信息：
{record}

请输出原子事实列表："""
        response = self.model.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        lines = response.strip().split("\n")
        for line in lines:
            line = line.strip()
            if line and line[0].isdigit():
                fact = line.split(".", 1)[-1].strip()
                if fact:
                    self.atomic_facts.append(fact)
        if not self.atomic_facts:
            self.atomic_facts = [record[:500]] if record else ["I came to see the doctor today."]

    def respond(self, doctor_message: str) -> str:
        relevant = self._select_relevant_facts(doctor_message)
        if not relevant:
            answer = "我不太清楚，没有特别注意到这个。"
        else:
            facts_text = "\n".join(f"- {f}" for f in relevant)
            prompt = f"""基于以下关于这个病人的事实，用普通病人的语言回答医生的问题。
只使用给出的事实，不要推测或添加信息。

病人事实：
{facts_text}

医生问题：{doctor_message}

病人回答（用第一人称，自然语言）："""
            answer = self.model.chat(
                [{"role": "user", "content": prompt}],
                system_prompt=self.SYSTEM_PROMPT,
                temperature=0.3,
            )

        self.conversation_history.append({"role": "doctor", "content": doctor_message})
        self.conversation_history.append({"role": "patient", "content": answer})
        return answer

    def _select_relevant_facts(self, question: str) -> List[str]:
        if not self.atomic_facts:
            return []
        if self.model.provider == "mock":
            return self.atomic_facts[:2]

        facts_numbered = "\n".join(f"{i+1}. {f}" for i, f in enumerate(self.atomic_facts))
        prompt = f"""从以下病人事实列表中，选出能回答医生问题的事实编号。
最多选2个，如果没有相关事实请回答"无"。
只输出编号，用逗号分隔（例如：1,3）或"无"。

事实列表：
{facts_numbered}

医生问题：{question}

相关事实编号："""
        response = self.model.chat([{"role": "user", "content": prompt}], temperature=0.0).strip()
        low = response.lower()
        if low in ("无", "none", "n/a", ""):
            return []

        selected: List[str] = []
        for part in response.split(","):
            try:
                idx = int(part.strip()) - 1
                if 0 <= idx < len(self.atomic_facts):
                    selected.append(self.atomic_facts[idx])
            except ValueError:
                continue
        return selected

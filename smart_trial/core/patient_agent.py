"""
Patient simulator: MediQ Fact-Select style (atomic facts + relevance selection).
Compatible with precomputed `atomic_facts` from local JSONL loader.
"""

from typing import Any, Dict, List

from smart_trial.models.model_client import ModelClient


class PatientAgent:
    SYSTEM_PROMPT = """You are a patient in a clinic visit. Answer only from the facts you have been given about your case.

Rules:
1. Answer only what the doctor asked; do not volunteer extra information.
2. Use simple everyday English, not medical jargon.
3. If your facts do not cover what they asked, say you are not sure or have not noticed.
4. Do not guess or make up information.
5. Reply only in English, even if the doctor accidentally uses another language."""

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

        prompt = f"""Split the following patient information into a numbered list of atomic facts.
Each fact must be one self-contained information item (one point per line).
Number each line like 1. ... 2. ...

Patient information:
{record}

Output only the numbered fact list in English:"""
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
            answer = "I'm not sure — I haven't really noticed that."
        else:
            facts_text = "\n".join(f"- {f}" for f in relevant)
            prompt = f"""Using only the facts below, answer the doctor's question in plain first-person English, as a lay patient would.
Do not add information that is not in the facts.

Facts you know:
{facts_text}

Doctor's question: {doctor_message}

Your answer (first person, English only):"""
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
        prompt = f"""From the numbered facts below, pick the indices (1-based) of facts needed to answer the doctor's question.
Pick at most 2 indices. If none apply, reply with exactly: none

Output format: comma-separated numbers only (e.g. 1,3) or none

Facts:
{facts_numbered}

Doctor question: {question}

Indices or none:"""
        response = self.model.chat([{"role": "user", "content": prompt}], temperature=0.0).strip()
        low = response.lower()
        if low in ("none", "n/a", "") or low.startswith("none"):
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

"""
Patient simulator: MediQ Fact-Select style (atomic facts + relevance selection).
Optional PatientPersona (see core/persona.py) modifies HOW the patient communicates
without changing WHAT they know — Fact-Select factuality is preserved.
"""

import random
import re
from typing import Any, Dict, List, Optional

from smart_trial.core.persona import PatientPersona
from smart_trial.models.model_client import ModelClient

_BASE_RULES = """Rules:
1. Answer only what the doctor asked; do not volunteer extra information.
2. Use simple everyday English, not medical jargon.
3. If your facts do not cover what they asked, say you are not sure or have not noticed.
4. Do not guess or make up information.
5. Reply only in English, even if the doctor accidentally uses another language."""

_OBJECTIVE_QUESTION_RE = re.compile(
    r"\b("
    r"lab|laboratory|blood\s+test|culture|imaging|x-?ray|ct\s+scan|mri|ultrasound|"
    r"urinalysis|biopsy|ekg|ecg|glucose|creatinine|hemoglobin|wbc|platelet|"
    r"physical\s+exam|examination\s+reveals|vital\s+signs?|blood\s+pressure|"
    r"temperature|pulse|respiratory\s+rate|oxygen\s+saturation|"
    r"pco2|po2|ph\b|bicarbonate|potassium|sodium|"
    r"antibiotic|prescri|treatment\s+given|injection|iv\s+fluid"
    r")\b",
    re.IGNORECASE,
)


def is_objective_clinical_question(question: str) -> bool:
    """Heuristic: labs, vitals, exam, and treatment facts → oracle (persona bypass)."""
    return bool(_OBJECTIVE_QUESTION_RE.search(question or ""))


def parse_age_years(age: Any) -> Optional[int]:
    """Extract patient age in years from loader fields like '21 years old' or 21."""
    if age is None:
        return None
    if isinstance(age, int):
        return age if 0 < age < 120 else None
    text = str(age).strip()
    if text.isdigit():
        value = int(text)
        return value if 0 < value < 120 else None
    match = re.search(r"(\d{1,3})", text)
    if match:
        value = int(match.group(1))
        return value if 0 < value < 120 else None
    return None


def parent_age_for_child(child_age: int) -> int:
    """Pick a plausible parent age in [30, 45] from the child's age."""
    return min(45, max(30, child_age + 28))


def age_voice_band(age_years: Optional[int]) -> str:
    if age_years is None:
        return "adult"
    if age_years < 13:
        return "child_parent"
    if age_years <= 17:
        return "teen"
    if age_years >= 65:
        return "older_adult"
    return "adult"


def _child_reference_word(gender: str) -> str:
    g = (gender or "unknown").lower()
    if g == "female":
        return "daughter"
    if g == "male":
        return "son"
    return "child"


def communication_cues(case: Dict[str, Any]) -> List[str]:
    """Detect special communication contexts from case text and return behaviour cues."""
    record = case.get("full_record") or case.get("chief_complaint") or ""
    cues: List[str] = []

    if re.search(
        r'\bsexual(?:ly)?\b|\bSTI\b|\bSTD\b|\burination\b|\bgenital\b|\bintercourse\b',
        record, re.IGNORECASE,
    ):
        cues.append(
            "Be comfortable sharing sexual-health details; avoid assumptions or stereotypes."
        )

    if re.search(
        r'\baccuse[sd]?\b|\bparanoi\b|\bhostile\b|\bdelusional\b|\bhallucinat\b',
        record, re.IGNORECASE,
    ):
        cues.append("You may sound guarded or suspicious of the doctor's intentions.")

    if re.search(
        r'\bworks?\s+as\b|\bemployed\b|\blogger\b|\boccupation\b|\bjob\b|\bprofession\b',
        record, re.IGNORECASE,
    ):
        cues.append("When relevant, naturally mention your work or school situation.")

    return cues


def build_patient_system_prompt(
    case: Dict[str, Any],
    persona: Optional[PatientPersona] = None,
) -> str:
    age_years = parse_age_years(case.get("age"))
    band = age_voice_band(age_years)
    gender = case.get("gender") or "unknown"

    if band == "child_parent":
        child_age = age_years if age_years is not None else 8
        parent_age = parent_age_for_child(child_age)
        child_word = _child_reference_word(gender)
        role = (
            f"You are a {parent_age}-year-old parent or guardian speaking on behalf of your "
            f"{child_age}-year-old {child_word}.\n"
            "Answer about your child's symptoms and history (say she/he/my child as appropriate), "
            "not in the child's voice.\n"
            "Use warm, concerned parent language in everyday English — clear and simple, not clinical."
        )
    elif band == "teen":
        role = (
            f"You are a {age_years}-year-old teenager in a clinic visit.\n"
            "Use natural, casual teen English (still polite and clear). Short sentences are fine. "
            "Avoid stiff, overly formal phrases."
        )
    elif band == "older_adult":
        role = (
            f"You are a {age_years}-year-old patient in a clinic visit.\n"
            "Use clear everyday English; you may sound slightly more formal or thoughtful than a "
            "young adult, but still plain language — not medical jargon."
        )
    elif age_years is not None:
        role = (
            f"You are a {age_years}-year-old adult patient in a clinic visit.\n"
            "Use normal, everyday conversational English."
        )
    else:
        role = (
            "You are an adult patient in a clinic visit.\n"
            "Use normal, everyday conversational English."
        )

    cues = communication_cues(case)
    cues_section = ""
    if cues:
        cues_text = "\n".join(f"- {c}" for c in cues)
        cues_section = f"\n\nCommunication notes:\n{cues_text}"

    persona_section = ""
    if persona is not None:
        block = persona.render_persona_block()
        if block:
            persona_section = f"\n\n{block}"

    return (
        f"{role}\n\n"
        "Answer only from the facts you have been given about this case.\n\n"
        f"{_BASE_RULES}"
        f"{cues_section}"
        f"{persona_section}"
    )


def default_uncertain_reply(
    case: Dict[str, Any],
    persona: Optional[PatientPersona] = None,
    last_reply: Optional[str] = None,
) -> str:
    age_years = parse_age_years(case.get("age"))
    band = age_voice_band(age_years)
    if band == "child_parent":
        return "I'm not sure — we haven't really noticed that about our child."
    if band == "teen":
        pool = [
            "Um, I'm not really sure — I haven't noticed that.",
            "Honestly no idea.",
            "I don't really know.",
        ]
        options = [r for r in pool if r != last_reply] or pool
        return random.choice(options)
    if persona is not None:
        return persona.uncertain_reply(last_reply=last_reply)
    replies = ["I'm not sure — I haven't really noticed that.", "I'm not certain about that."]
    options = [r for r in replies if r != last_reply] or replies
    return random.choice(options)


def answer_style_instruction(case: Dict[str, Any]) -> str:
    band = age_voice_band(parse_age_years(case.get("age")))
    if band == "child_parent":
        return (
            "Answer as the parent or guardian on behalf of your child, in plain first-person English. "
            "Do not add information that is not in the facts."
        )
    return (
        "Answer in plain first-person English, as this patient would. "
        "Do not add information that is not in the facts."
    )


class PatientAgent:
    """Fact-Select patient simulator with age-appropriate voice and optional persona."""

    def __init__(
        self,
        model_client: ModelClient,
        case: Dict[str, Any],
        persona: Optional[PatientPersona] = None,
    ):
        self.model = model_client
        self.case = case
        self.persona = persona
        self.system_prompt = build_patient_system_prompt(case, persona=persona)
        self.atomic_facts: List[str] = []
        self.conversation_history: List[Dict[str, str]] = []
        self._last_uncertain_reply: Optional[str] = None
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
        if is_objective_clinical_question(doctor_message):
            answer = self._respond_oracle(doctor_message)
            self.conversation_history.append({"role": "doctor", "content": doctor_message})
            self.conversation_history.append({"role": "patient", "content": answer})
            return answer

        # Jargon short-circuit: patient asks for clarification instead of answering
        if self.persona is not None:
            term = self.persona.detects_jargon(doctor_message)
            if term is not None:
                prompt = self.persona.clarification_prompt(term)
                answer = self.model.chat(
                    [{"role": "user", "content": prompt}],
                    system_prompt=self.system_prompt,
                    temperature=0.5,
                )
                self.conversation_history.append({"role": "doctor", "content": doctor_message})
                self.conversation_history.append({"role": "patient", "content": answer})
                return answer

        relevant = self._select_relevant_facts(doctor_message)
        if not relevant:
            answer = default_uncertain_reply(
                self.case, persona=self.persona, last_reply=self._last_uncertain_reply
            )
            self._last_uncertain_reply = answer
        else:
            facts_text = "\n".join(f"- {f}" for f in relevant)
            style = answer_style_instruction(self.case)
            prompt = f"""Using only the facts below, answer the doctor's question.
{style}
Stay in character throughout your response.

Facts you know:
{facts_text}

Doctor's question: {doctor_message}

Your answer (English only):"""
            answer = self.model.chat(
                [{"role": "user", "content": prompt}],
                system_prompt=self.system_prompt,
                temperature=0.3,
            )

        self.conversation_history.append({"role": "doctor", "content": doctor_message})
        self.conversation_history.append({"role": "patient", "content": answer})
        return answer

    def _respond_oracle(self, doctor_message: str) -> str:
        """Objective clinical data: give atomic fact content directly (bypass persona)."""
        relevant = self._select_relevant_facts(doctor_message)
        if not relevant:
            if self.model.provider == "mock" and self.atomic_facts:
                relevant = [self.atomic_facts[0]]
            else:
                return default_uncertain_reply(
                    self.case, persona=None, last_reply=self._last_uncertain_reply
                )
        if self.model.provider == "mock":
            return relevant[0]
        facts_text = "\n".join(f"- {f}" for f in relevant)
        prompt = f"""You are a clinical data oracle. The doctor asked a question about objective
clinical information (labs, vitals, exam, or treatments). Answer using ONLY the facts below.
State the fact clearly in plain English. Do NOT use persona quirks, evasion, or uncertainty
if the facts contain the answer.

Facts:
{facts_text}

Doctor's question: {doctor_message}

Your answer (English only):"""
        return self.model.chat(
            [{"role": "user", "content": prompt}],
            system_prompt=(
                "You relay exact clinical facts from the chart. Be direct and factual. "
                "Ignore personality instructions."
            ),
            temperature=0.1,
        )

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

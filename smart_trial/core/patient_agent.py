"""
Patient simulator: MediQ Fact-Select style (atomic facts + relevance selection).
Compatible with precomputed `atomic_facts` from local JSONL loader.

Optional health-literacy persona (see core/persona.py) wraps this agent with
configurable communication behavior. When a persona is attached:
  - The persona's instruction block is appended to the system prompt.
  - Doctor messages are first passed through a jargon-comprehension check.
    If the patient's literacy level cannot parse a term in the question, the
    Fact-Select pipeline is bypassed and a clarification request is returned
    in the persona's voice.
Factuality is preserved because the persona only changes how the patient
communicates -- it never modifies the atomic-fact retrieval or generation.
"""

<<<<<<< HEAD
import re
from typing import Any, Dict, List, Optional

from smart_trial.core.persona import HealthLiteracyPersona
from smart_trial.core.trust_disclosure_persona import TrustDisclosurePersona
=======
from typing import Any, Dict, List, Optional

from smart_trial.core.persona import HealthLiteracyPersona
>>>>>>> b09bbc8 (Add health-literacy patient persona)
from smart_trial.models.model_client import ModelClient

_BASE_RULES = """Rules:
1. Answer only what the doctor asked; do not volunteer extra information.
2. Use simple everyday English, not medical jargon.
3. If your facts do not cover what they asked, say you are not sure or have not noticed.
4. Do not guess or make up information.
5. Reply only in English, even if the doctor accidentally uses another language."""

<<<<<<< HEAD

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


def build_patient_system_prompt(case: Dict[str, Any]) -> str:
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

    return (
        f"{role}\n\n"
        "Answer only from the facts you have been given about this case.\n\n"
        f"{_BASE_RULES}"
    )


def default_uncertain_reply(case: Dict[str, Any]) -> str:
    age_years = parse_age_years(case.get("age"))
    band = age_voice_band(age_years)
    if band == "child_parent":
        return "I'm not sure — we haven't really noticed that about our child."
    if band == "teen":
        return "Um, I'm not really sure — I haven't noticed that."
    if band == "older_adult":
        return "I'm afraid I'm not certain — I haven't noticed that, honestly."
    return "I'm not sure — I haven't really noticed that."


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
    """Fact-Select patient simulator with age-appropriate voice in the system prompt.

    Two optional persona layers, both additive and orthogonal:
      - `literacy_persona` (see core/persona.py) controls HOW the patient
        communicates: vocabulary register, jargon comprehension, anatomical
        localization. Adds a jargon short-circuit -- when the doctor uses a
        term the patient doesn't understand, the encounter returns a
        clarification request instead of running Fact-Select.
      - `trust_persona` (see core/trust_disclosure_persona.py) controls
        WHAT the patient is willing to share: trust level, concealed topics,
        reaction to insensitive questioning. Adds an opener-aware fact
        filter -- facts on concealed topics are suppressed unless the doctor
        opens with normalizing / patient-centered phrasing.
    Factuality is preserved: neither persona modifies the underlying atomic
    facts; both only change how those facts are surfaced.
    """

=======
>>>>>>> b09bbc8 (Add health-literacy patient persona)
    def __init__(
        self,
        model_client: ModelClient,
        case: Dict[str, Any],
<<<<<<< HEAD
        literacy_persona: Optional[HealthLiteracyPersona] = None,
        trust_persona: Optional[TrustDisclosurePersona] = None,
    ):
        self.model = model_client
        self.case = case
        self.literacy_persona = literacy_persona
        self.trust_persona = trust_persona
        self.system_prompt = build_patient_system_prompt(case)
=======
        persona: Optional[HealthLiteracyPersona] = None,
    ):
        self.model = model_client
        self.case = case
        self.persona = persona
>>>>>>> b09bbc8 (Add health-literacy patient persona)
        self.atomic_facts: List[str] = []
        self.conversation_history: List[Dict[str, str]] = []
        self._init_facts()

    def _effective_system_prompt(self) -> str:
<<<<<<< HEAD
        """Per-call system prompt including any attached persona blocks."""
        parts: List[str] = [self.system_prompt]
        if self.literacy_persona is not None:
            parts.append(self.literacy_persona.render_persona_block())
        if self.trust_persona is not None:
            parts.append(self.trust_persona.render_persona_block())
        return "\n\n".join(parts)
=======
        """System prompt with the persona instruction block appended (if any)."""
        if self.persona is None:
            return self.SYSTEM_PROMPT
        return self.SYSTEM_PROMPT + "\n\n" + self.persona.render_persona_block()
>>>>>>> b09bbc8 (Add health-literacy patient persona)

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
<<<<<<< HEAD
        # Literacy persona jargon short-circuit: if the patient cannot parse
        # a term in the doctor's question, skip Fact-Select and return a
        # clarification request in the persona's voice. Models real
        # low-literacy patients who ask "what does that mean?" first.
        if self.literacy_persona is not None:
            unknown_term = self.literacy_persona.detects_jargon(doctor_message)
            if unknown_term is not None:
                answer = self.literacy_persona.clarification_request(unknown_term)
=======
        # Persona jargon short-circuit: if the persona cannot parse a term in
        # the doctor's question, skip Fact-Select and emit a clarification
        # request in the persona's voice. This models real low-literacy
        # patients who ask "what does that mean?" before answering.
        if self.persona is not None:
            unknown_term = self.persona.detects_jargon(doctor_message)
            if unknown_term is not None:
                answer = self.persona.clarification_request(unknown_term)
>>>>>>> b09bbc8 (Add health-literacy patient persona)
                self.conversation_history.append({"role": "doctor", "content": doctor_message})
                self.conversation_history.append({"role": "patient", "content": answer})
                return answer

        relevant = self._select_relevant_facts(doctor_message)

        # Trust & Disclosure persona: filter out concealed-topic facts unless
        # the doctor's question contained a trust-respecting opener. If the
        # filter empties the selection, the patient declines to disclose and
        # the default uncertain reply is used -- modeling the wary patient
        # who answers "not really" instead of surfacing the hidden topic.
        if self.trust_persona is not None and relevant:
            relevant = self.trust_persona.filter_facts(relevant, doctor_message)

        if not relevant:
            answer = default_uncertain_reply(self.case)
        else:
            facts_text = "\n".join(f"- {f}" for f in relevant)
            style = answer_style_instruction(self.case)
            prompt = f"""Using only the facts below, answer the doctor's question.
{style}

Facts you know:
{facts_text}

Doctor's question: {doctor_message}

Your answer (English only):"""
            answer = self.model.chat(
                [{"role": "user", "content": prompt}],
                system_prompt=self._effective_system_prompt(),
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

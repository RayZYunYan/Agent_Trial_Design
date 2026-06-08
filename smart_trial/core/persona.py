"""
Unified patient persona for SMART Trial patient simulator.

Five axes:
  vocabulary_register   : functional/interactive/critical  — how the patient expresses themselves
  jargon_comprehension  : functional/interactive/critical  — what medical terms the patient understands
  personality           : neutral/anxious/impatient/reserved/verbose/distrustful
  recall                : high/low — certainty of dates/amounts (underlying fact preserved)
  emotional_state       : calm/distressed/evasive

Health literacy levels follow Nutbeam (2000):
  functional  — basic reading/comprehension, everyday language only
  interactive — common medical terms understood, but not specialist vocabulary
  critical    — health-literate, may research conditions and use clinical terminology
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# ---------------------------------------------------------------------------
# Valid axis values — used for sampling and validation
# ---------------------------------------------------------------------------
PERSONA_AXES: Dict[str, List[str]] = {
    "vocabulary_register":  ["functional", "interactive", "critical"],
    "jargon_comprehension": ["functional", "interactive", "critical"],
    "personality":          ["neutral", "anxious", "impatient", "reserved", "verbose", "distrustful"],
    "recall":               ["high", "low"],
    "emotional_state":      ["calm", "distressed", "evasive"],
}

# ---------------------------------------------------------------------------
# Jargon term lists (Nutbeam health literacy model)
# _SPECIALIST_TERMS      : interactive and functional patients don't know these
# _COMMON_MEDICAL_TERMS  : only functional patients don't know these
# ---------------------------------------------------------------------------
_SPECIALIST_TERMS = (
    "palpitation", "palpitations",
    "syncope", "presyncope",
    "dyspnea", "dyspnoea",
    "orthopnea",
    "tachycardia", "bradycardia",
    "diaphoresis",
    "claudication",
    "paresthesia", "paraesthesia",
    "dysphagia",
    "hemoptysis",
    "hematuria",
    "edema", "oedema",
)

_COMMON_MEDICAL_TERMS = (
    "radiation", "radiating", "radiate",
    "review of systems",
    "chief complaint",
    "associated symptoms",
    "exacerbating", "alleviating",
    "aggravating", "relieving",
    "onset", "duration", "severity",
    "constitutional symptoms",
    "differential",
    "contraindication",
)

# ---------------------------------------------------------------------------
# Per-personality instruction text (injected into system prompt)
# ---------------------------------------------------------------------------
_PERSONALITY_INSTRUCTIONS: Dict[str, str] = {
    "anxious":
        "You are visibly anxious. Express worry freely and seek reassurance, "
        "e.g. \"Is that serious?\" or \"Should I be worried about that?\"",
    "impatient":
        "You want to finish the visit quickly. Keep answers short and to the point. "
        "Show mild impatience if the doctor asks too many questions.",
    "reserved":
        "You are reluctant to volunteer information. Only answer exactly what is asked "
        "— make the doctor work to draw out details.",
    "verbose":
        "You tend to ramble. Include extra life context and side details even when "
        "not directly relevant to the question.",
    "distrustful":
        "You are skeptical of the doctor. Question their motives when asked sensitive "
        "questions, e.g. \"Why does that matter?\" or \"Why do you need to know that?\"",
}

# ---------------------------------------------------------------------------
# Per-personality uncertain replies (used when no relevant facts found)
# ---------------------------------------------------------------------------
_UNCERTAIN_REPLIES: Dict[str, List[str]] = {
    "neutral":     ["I'm not sure — I haven't really noticed that.", "I'm not certain about that."],
    "anxious":     ["I'm not sure... is that something I should have noticed?", "I don't know — should I be worried?"],
    "impatient":   ["I don't know.", "Not sure."],
    "reserved":    ["I don't know.", "I haven't noticed."],
    "verbose":     [
        "Honestly I'm not really sure. I've been so busy lately I haven't paid close attention to that.",
        "I don't know — I mean, there's been a lot going on, I haven't been keeping track.",
    ],
    "distrustful": ["I don't know. Why does that matter?", "Not sure. Why are you asking about that?"],
}


@dataclass
class PatientPersona:
    persona_id: str = "default"
    vocabulary_register: str = "interactive"
    jargon_comprehension: str = "interactive"
    personality: str = "neutral"
    recall: str = "high"
    emotional_state: str = "calm"

    # ------------------------------------------------------------------
    # System prompt block
    # ------------------------------------------------------------------
    def render_persona_block(self) -> str:
        """Return instruction text to append to the patient system prompt."""
        lines: List[str] = []

        if self.vocabulary_register == "functional":
            lines.append(
                "Use very simple everyday words and short sentences. "
                "Say things like \"my stomach hurts bad\" rather than \"abdominal pain\". "
                "Never use medical terminology."
            )
        elif self.vocabulary_register == "critical":
            lines.append(
                "You are articulate and well-informed. You may use terms you looked up, "
                "e.g. \"I read it could be gastritis\" or \"my blood pressure has been elevated\"."
            )

        if self.personality in _PERSONALITY_INSTRUCTIONS:
            lines.append(_PERSONALITY_INSTRUCTIONS[self.personality])

        if self.recall == "low":
            lines.append(
                "You are uncertain about exact dates, durations, and dosages. "
                "Add hedging language like \"I think it was...\" or \"around that time\" — "
                "but still state the underlying fact; do not omit or change it."
            )

        if self.emotional_state == "distressed":
            lines.append(
                "You feel stressed and overwhelmed. Occasionally mention how this affects "
                "your daily life, e.g. \"I haven't been able to sleep because of this.\""
            )
        elif self.emotional_state == "evasive":
            lines.append(
                "You are uncomfortable with certain topics. Hesitate before answering "
                "sensitive questions, but ultimately provide the information if pressed."
            )

        if not lines:
            return ""
        return "Persona — follow these communication rules:\n" + "\n".join(f"- {l}" for l in lines)

    # ------------------------------------------------------------------
    # Jargon short-circuit (called in PatientAgent.respond)
    # ------------------------------------------------------------------
    def detects_jargon(self, doctor_message: str) -> Optional[str]:
        """Return the first unrecognised term in doctor_message, or None."""
        if self.jargon_comprehension == "critical":
            return None
        text = doctor_message.lower()
        triggers = list(_SPECIALIST_TERMS)
        if self.jargon_comprehension == "functional":
            triggers += list(_COMMON_MEDICAL_TERMS)
        for term in triggers:
            if term in text:
                return term
        return None

    def clarification_prompt(self, term: str) -> str:
        """Return a prompt for the LLM to generate a natural clarification request."""
        return (
            f"The doctor just used the term \"{term}\" which you do not understand. "
            "Ask the doctor to explain it in simpler terms. Stay in character."
        )

    # ------------------------------------------------------------------
    # Uncertain reply (used when Fact-Select finds no relevant facts)
    # ------------------------------------------------------------------
    def uncertain_reply(self, last_reply: Optional[str] = None) -> str:
        replies = _UNCERTAIN_REPLIES.get(self.personality, _UNCERTAIN_REPLIES["neutral"])
        options = [r for r in replies if r != last_reply] or replies
        return random.choice(options)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_id":           self.persona_id,
            "vocabulary_register":  self.vocabulary_register,
            "jargon_comprehension": self.jargon_comprehension,
            "personality":          self.personality,
            "recall":               self.recall,
            "emotional_state":      self.emotional_state,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PatientPersona":
        return cls(
            persona_id=str(d.get("persona_id", "default")),
            vocabulary_register=str(d.get("vocabulary_register", "interactive")),
            jargon_comprehension=str(d.get("jargon_comprehension", "interactive")),
            personality=str(d.get("personality", "neutral")),
            recall=str(d.get("recall", "high")),
            emotional_state=str(d.get("emotional_state", "calm")),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "PatientPersona":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        axes = cfg.get("axes") or cfg
        axes["persona_id"] = cfg.get("persona_id", path.stem)
        return cls.from_dict(axes)

    @classmethod
    def sample(cls, rng: random.Random) -> "PatientPersona":
        """Randomly sample one persona — reproducible given the same rng."""
        return cls(
            persona_id="random",
            vocabulary_register=rng.choice(PERSONA_AXES["vocabulary_register"]),
            jargon_comprehension=rng.choice(PERSONA_AXES["jargon_comprehension"]),
            personality=rng.choice(PERSONA_AXES["personality"]),
            recall=rng.choice(PERSONA_AXES["recall"]),
            emotional_state=rng.choice(PERSONA_AXES["emotional_state"]),
        )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_persona_from_id(
    persona_id: str,
    personas_dir: Optional[Path] = None,
) -> PatientPersona:
    if personas_dir is None:
        personas_dir = Path(__file__).resolve().parent.parent / "config" / "personas"
    path = personas_dir / f"{persona_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Persona YAML not found: {path}")
    return PatientPersona.from_yaml(path)


def select_default_persona_id(case: Dict[str, Any]) -> str:
    """Demographic default: younger/middle-aged adults → default, children/elderly → low_literacy."""
    age_raw = case.get("age")
    try:
        match = re.search(r"\d+", str(age_raw)) if age_raw else None
        age = int(match.group()) if match else None
    except (AttributeError, ValueError):
        age = None
    if age is None:
        return "default"
    if age < 18 or age >= 70:
        return "low_literacy"
    return "default"

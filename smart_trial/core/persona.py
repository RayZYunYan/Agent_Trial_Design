"""
Health-Literacy Persona module for the SMART Trial patient simulator.

Wraps the existing MediQ Fact-Select PatientAgent with a configurable persona
that modifies (a) how the patient comprehends doctor jargon and (b) the
linguistic register of the patient's reply, without changing what facts the
patient knows or how those facts are selected. Factuality is preserved because
the persona only affects how an already-grounded answer is rendered -- it never
re-reads case facts or invents new ones.

MVP axes (v1, per spec):
- vocabulary_register      F (lay) / I (mixed) / C (clinical)
- jargon_comprehension     F (silently misinterpret or ask) / I (partial) / C (full)
- anatomical_localization  F (vague) / I (rough zone) / C (precise region)

Levels map to Nutbeam (2000) three-level health-literacy framework:
    F = functional  (basic)
    I = interactive (communicative)
    C = critical    (advanced)

Loaded from YAML in smart_trial/config/personas/*.yaml.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

LEVELS = ("F", "I", "C")
_AXES = ("vocabulary_register", "jargon_comprehension", "anatomical_localization")


@dataclass
class HealthLiteracyPersona:
    persona_id: str = "literacy_C"
    nutbeam_level: str = "critical"
    vocabulary_register: str = "C"
    jargon_comprehension: str = "C"
    anatomical_localization: str = "C"
    notes: str = ""

    # ----- construction -----

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "HealthLiteracyPersona":
        axes = cfg.get("axes") or {}
        for axis in _AXES:
            val = axes.get(axis)
            if val not in LEVELS:
                raise ValueError(
                    f"Persona axis {axis!r} must be one of {LEVELS}, got {val!r} "
                    f"in persona_id={cfg.get('persona_id')!r}"
                )
        return cls(
            persona_id=str(cfg.get("persona_id", "literacy_custom")),
            nutbeam_level=str(cfg.get("nutbeam_level", "mixed")),
            vocabulary_register=axes["vocabulary_register"],
            jargon_comprehension=axes["jargon_comprehension"],
            anatomical_localization=axes["anatomical_localization"],
            notes=str(cfg.get("notes", "")),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "HealthLiteracyPersona":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"Persona YAML at {path} must be a mapping")
        return cls.from_dict(cfg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "nutbeam_level": self.nutbeam_level,
            "axes": {
                "vocabulary_register": self.vocabulary_register,
                "jargon_comprehension": self.jargon_comprehension,
                "anatomical_localization": self.anatomical_localization,
            },
        }

    # ----- prompt rendering -----

    _VOCAB_INSTRUCTIONS = {
        "F": (
            "Use everyday lay words ONLY. Never use medical terminology. "
            "Examples: say 'tummy' not 'abdomen', 'pee' not 'urinate', "
            "'throw up' not 'vomit', 'hurts when I breathe' not 'pleuritic', "
            "'really bad' not '8 out of 10'."
        ),
        "I": (
            "Use a mix of everyday and basic medical words. You know common "
            "terms (stomach, urinate, heart attack, blood pressure, infection) "
            "but not specialist ones."
        ),
        "C": (
            "You are comfortable with medical terminology. Use precise clinical "
            "words where they fit (epigastric, hematuria, palpitations, dyspnea)."
        ),
    }

    _JARGON_INSTRUCTIONS = {
        "F": (
            "If the doctor uses a medical term you don't understand (e.g. "
            "radiation, palpitations, syncope, dyspnea, edema, OPQRST), "
            "do NOT pretend to understand. Either ask 'what do you mean?' "
            "or say 'I'm not sure what that means.' Do not silently guess."
        ),
        "I": (
            "You understand common medical terms (heart attack, blood pressure, "
            "infection) but ask for clarification when the doctor uses specialist "
            "terms (palpitations, syncope, dyspnea, ischemia)."
        ),
        "C": (
            "You understand the doctor's medical terms and answer them directly "
            "as intended."
        ),
    }

    _ANATOMY_INSTRUCTIONS = {
        "F": (
            "Describe where it hurts with vague everyday language: 'around "
            "here', 'in my chest area', 'kinda in my belly', 'somewhere on "
            "the side'. Never use precise anatomical terms."
        ),
        "I": (
            "Locate sensations with rough zones: 'upper belly', 'left side of "
            "my chest', 'lower back', 'behind my eye'. Do not use medical "
            "quadrants or precise regions."
        ),
        "C": (
            "Locate sensations with precise anatomical language where it "
            "applies: 'right lower quadrant', 'substernal', 'between my "
            "shoulder blades', 'left flank'."
        ),
    }

    def render_persona_block(self) -> str:
        return "\n".join(
            [
                "Patient persona -- you MUST follow these communication rules:",
                f"- Vocabulary: {self._VOCAB_INSTRUCTIONS[self.vocabulary_register]}",
                f"- Jargon comprehension: {self._JARGON_INSTRUCTIONS[self.jargon_comprehension]}",
                f"- Locating sensations: {self._ANATOMY_INSTRUCTIONS[self.anatomical_localization]}",
                "",
                "These rules describe HOW you communicate. They never change WHAT "
                "you know -- do not invent new facts to fit the persona, and do not "
                "withhold facts you do know just because they sound technical.",
            ]
        )

    # ----- jargon comprehension check (heuristic, no API call) -----

    _SPECIALIST_TERMS = (
        "palpitation", "palpitations",
        "syncope", "presyncope",
        "dyspnea", "dyspnoea",
        "edema", "oedema",
        "hemoptysis", "haemoptysis",
        "ischemia", "ischemic", "ischaemia",
        "epigastric", "substernal", "pleuritic",
        "hematuria", "haematuria",
        "hematochezia", "haematochezia",
        "melena", "melaena",
        "diaphoresis", "diaphoretic",
        "orthopnea", "orthopnoea",
        "tachycardia", "bradycardia",
        "claudication",
        "paresthesia", "paraesthesia",
        "dysphagia",
    )

    _COMMON_MEDICAL_TERMS = (
        "radiation", "radiating", "radiate",
        "review of systems", "ros",
        "opqrst",
        "differential diagnosis",
        "presenting complaint",
        "chief complaint",
        "associated symptoms",
        "exacerbating", "alleviating",
        "onset",
        "duration",
        "severity",
        "aggravating", "relieving",
        "constitutional symptoms",
    )

    def detects_jargon(self, doctor_message: str) -> Optional[str]:
        if self.jargon_comprehension == "C":
            return None
        text = (doctor_message or "").lower()
        triggers = list(self._SPECIALIST_TERMS)
        if self.jargon_comprehension == "F":
            triggers += list(self._COMMON_MEDICAL_TERMS)
        for term in triggers:
            if term in text:
                return term
        return None

    def clarification_request(self, term: str) -> str:
        if self.vocabulary_register == "F":
            return (
                f"I'm not sure what \"{term}\" means -- could you say that "
                f"in a simpler way?"
            )
        if self.vocabulary_register == "I":
            return f"Sorry, what do you mean by \"{term}\"?"
        return f"Could you clarify what you mean by \"{term}\"?"


# ----- loaders & defaults --------------------------------------------------


def load_persona_from_id(
    persona_id: str,
    personas_dir: Optional[Path] = None,
) -> HealthLiteracyPersona:
    if personas_dir is None:
        personas_dir = Path(__file__).resolve().parent.parent / "config" / "personas"
    path = personas_dir / f"{persona_id}.yaml"
    return HealthLiteracyPersona.from_yaml(path)


def select_default_persona_id(case: Dict[str, Any]) -> str:
    """
    Demographically-informed default persona for a case.

    NOTE: imports demographic stereotypes by design. For controlled
    experiments use `trial_config.persona.mode: fixed` to override.

    Heuristic:
        age < 18 or age >= 70  -> literacy_F
        age 18..69             -> literacy_I
        age unknown            -> literacy_I
    """
    age_raw = case.get("age")
    try:
        age = int(str(age_raw).strip()) if age_raw not in (None, "") else None
    except (ValueError, TypeError):
        age = None
    if age is None:
        return "literacy_I"
    if age < 18 or age >= 70:
        return "literacy_F"
    return "literacy_I"

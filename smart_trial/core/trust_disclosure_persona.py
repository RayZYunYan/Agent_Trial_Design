"""
Trust & Disclosure Persona (TDP) for the SMART Trial patient simulator.

Models a configurable patient stance toward the clinician and a configurable
set of stigmatized topics the patient conceals unless asked with a
trust-respecting opener. Composes additively with the HealthLiteracyPersona:
literacy controls HOW the patient talks; TDP controls WHAT the patient
chooses to share and how they react to insensitive questioning.

Axes (MVP v1):
- trust_in_clinician           low | moderate | high
- concealed_topics             list of topic strings (default empty)
                               standard menu: alcohol, drug_use,
                               sexual_history, mental_health,
                               intimate_partner_violence, non_adherence,
                               prior_negative_medical_experience, firearms
- reaction_to_insensitive      withdrawal | guarded | persistent

Theoretical anchors:
- Trust in Physician Scale (Anderson & Dedrick 1990)
- Medical Mistrust Index (LaVeist et al. 2009)
- Hidden-agenda / disclosure literature (Barry et al.; Bass & Cohen)

This module is deliberately prompt-driven plus a deterministic
substring/regex opener detector. No second LLM pass is required, so cost
matches the existing literacy persona and the detector is auditable in a
paper appendix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import re

import yaml

TRUST_LEVELS = ("low", "moderate", "high")
REACTION_LEVELS = ("withdrawal", "guarded", "persistent")

# Standard menu of concealable topics. Free-text additions are allowed -- this
# list just provides the keyword maps the fact filter consults.
STANDARD_TOPICS = (
    "alcohol",
    "drug_use",
    "sexual_history",
    "mental_health",
    "intimate_partner_violence",
    "non_adherence",
    "prior_negative_medical_experience",
    "firearms",
)


# ---------- keyword maps -----------------------------------------------------
#
# Substring matching on the lowercased fact text or doctor message. Lists
# below are intentionally conservative (high-precision); errors of omission
# are preferable to false positives that wrongly suppress facts the patient
# is happy to disclose.

_TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "alcohol": [
        "alcohol", "drink", "beer", "wine", "liquor",
        "drunk", "intoxicat", "binge",
    ],
    "drug_use": [
        " drug ", "cocaine", "heroin", "meth", "marijuana", "cannabis",
        " weed", " ivdu", "intravenous drug", "injection drug",
        "needle ", "opioid",
    ],
    "sexual_history": [
        " sex ", "sexual", "partner", "unprotected", "condom",
        " sti ", " std ", " hiv ", "gonorrh", "chlamydia",
    ],
    "mental_health": [
        "depress", "anxiet", "suicid", "psychiatric", "psychos",
        "therap", "antidepressant", "panic attack",
    ],
    "intimate_partner_violence": [
        "hit me", "hits me", "violence", " abus", "afraid of",
        "scared of", "domestic", "partner hurt",
    ],
    "non_adherence": [
        "stopped taking", "skipped", "haven't taken", "haven't been taking",
        "forgot medication", "ran out", "couldn't afford",
    ],
    "prior_negative_medical_experience": [
        "bad experience", "wasn't listened to", "dismissed me",
        "didn't believe me", "misdiagnos",
    ],
    "firearms": [
        "firearm", "gun ", " guns", "weapon", "pistol", "rifle",
    ],
}

# Phrases that count as a "trust-respecting opener" -- non-judgmental,
# normalizing, or invitationally open-ended. If any appears (case-insensitive
# substring) in the doctor's message, concealed-topic facts become eligible.
_OPENER_PHRASES: List[str] = [
    # normalizing
    "many patients",
    "it's common",
    "it is common",
    "lots of people",
    "no judgment",
    "no judgement",
    "this is confidential",
    "won't be judged",
    # open-ended invitation
    "can you tell me",
    "could you tell me",
    "would you mind sharing",
    "is there anything",
    "anything you're worried",
    "anything you are worried",
    "is there something",
    "what's important to you",
    "what is important to you",
    "what brings you",
    "how are things",
    # normalization framing
    "sometimes people",
    "some patients",
    "i ask everyone",
    "i ask all my patients",
]

# Phrases that count as forensic / judgmental: yes/no probes on stigmatized
# topics phrased in a way that suppresses disclosure. Used to trigger the
# reaction policy (withdrawal etc.).
_FORENSIC_PATTERNS: List[str] = [
    r"\bdo you abuse\b",
    r"\bdo you use drugs\b",
    r"\bhow much do you drink\b",
    r"\bare you sexually active\b",
    r"\bany iv drug use\b",
    r"\bany unprotected\b",
    r"\bdo you have unprotected\b",
]


# ---------- TDP dataclass -----------------------------------------------------


@dataclass
class TrustDisclosurePersona:
    persona_id: str = "cooperative"
    trust_in_clinician: str = "high"
    concealed_topics: List[str] = field(default_factory=list)
    reaction_to_insensitive: str = "persistent"
    notes: str = ""

    # ----- construction -----

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "TrustDisclosurePersona":
        axes = cfg.get("axes") or {}
        trust = axes.get("trust_in_clinician")
        if trust not in TRUST_LEVELS:
            raise ValueError(
                f"trust_in_clinician must be one of {TRUST_LEVELS}, "
                f"got {trust!r} in persona_id={cfg.get('persona_id')!r}"
            )
        reaction = axes.get("reaction_to_insensitive")
        if reaction not in REACTION_LEVELS:
            raise ValueError(
                f"reaction_to_insensitive must be one of {REACTION_LEVELS}, "
                f"got {reaction!r}"
            )
        concealed = axes.get("concealed_topics") or []
        if not isinstance(concealed, list):
            raise ValueError("concealed_topics must be a list of strings")
        concealed = [str(t).strip() for t in concealed if str(t).strip()]
        return cls(
            persona_id=str(cfg.get("persona_id", "trust_custom")),
            trust_in_clinician=trust,
            concealed_topics=concealed,
            reaction_to_insensitive=reaction,
            notes=str(cfg.get("notes", "")),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> "TrustDisclosurePersona":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"TDP YAML at {path} must be a mapping")
        return cls.from_dict(cfg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "persona_id": self.persona_id,
            "axes": {
                "trust_in_clinician": self.trust_in_clinician,
                "concealed_topics": list(self.concealed_topics),
                "reaction_to_insensitive": self.reaction_to_insensitive,
            },
        }

    # ----- prompt rendering -----

    _TRUST_INSTRUCTIONS = {
        "low": (
            "You do not trust this clinician yet. Give minimal answers. Do not "
            "volunteer information. If the clinician's question feels intrusive "
            "or rushed, you may push back ('why do you need to know that?') or "
            "give short, evasive answers. You may accept advice only when it is "
            "clearly explained."
        ),
        "moderate": (
            "You are reasonably willing to engage with this clinician but you "
            "verify what they say. You may ask 'are you sure?', 'what would tell "
            "you that?' or 'what test would show that?' before accepting advice."
        ),
        "high": (
            "You trust this clinician. Cooperate fully, answer directly, and "
            "accept their reasoning without push-back."
        ),
    }

    _REACTION_INSTRUCTIONS = {
        "withdrawal": (
            "If the clinician asks about a sensitive topic in a judgmental or "
            "interrogating way (e.g. 'do you abuse drugs?', 'how much do you "
            "drink?', 'are you sexually active?'), you SHUT DOWN. Your next two "
            "answers become minimal ('I don't really want to talk about that', "
            "'not much') and you do NOT disclose the concealed information even "
            "if you have it."
        ),
        "guarded": (
            "If the clinician asks about a sensitive topic in a judgmental or "
            "interrogating way, you deflect to a safer topic and avoid the "
            "concealed information for the rest of the turn."
        ),
        "persistent": (
            "Even when the clinician's phrasing is awkward or judgmental, you "
            "answer truthfully from the facts you have."
        ),
    }

    def render_persona_block(self) -> str:
        topics_clause = (
            "You have NO topics you are hiding from the clinician."
            if not self.concealed_topics
            else (
                "You are hiding the following sensitive topics from the "
                "clinician unless they ask about them in a non-judgmental, "
                "patient-centered, normalizing way: "
                + ", ".join(self.concealed_topics)
                + ". Even when you have facts about these topics, do NOT "
                "volunteer them until the clinician opens an invitation that "
                "feels safe (e.g. 'many patients in your situation...', "
                "'is there anything you're worried about?', 'no judgment, but...'"
                "). If the clinician asks bluntly or judgmentally, follow your "
                "reaction policy below instead of disclosing."
            )
        )
        return "\n".join(
            [
                "Patient stance toward the clinician — you MUST follow these rules:",
                f"- Trust level: {self._TRUST_INSTRUCTIONS[self.trust_in_clinician]}",
                f"- Concealed topics: {topics_clause}",
                f"- Reaction to insensitive questioning: "
                f"{self._REACTION_INSTRUCTIONS[self.reaction_to_insensitive]}",
                "",
                "These rules describe your stance and what you choose to share. "
                "They never change WHAT facts you actually have — do not invent "
                "facts to fit the persona.",
            ]
        )

    # ----- opener / forensic detection ------------------------------------------

    def doctor_used_opener(self, doctor_message: str) -> bool:
        """Does the doctor's message contain a trust-respecting opener phrase?"""
        text = (doctor_message or "").lower()
        return any(phrase in text for phrase in _OPENER_PHRASES)

    def doctor_was_forensic(self, doctor_message: str) -> bool:
        """Did the doctor use judgmental / forensic phrasing? (Triggers reaction.)"""
        text = (doctor_message or "").lower()
        return any(re.search(p, text) for p in _FORENSIC_PATTERNS)

    # ----- fact-level concealment ----------------------------------------------

    @staticmethod
    def _fact_matches_topic(fact: str, topic: str) -> bool:
        """Does this fact look like it's about `topic`? (Keyword substring match.)"""
        text = (" " + (fact or "").lower() + " ")
        keywords = _TOPIC_KEYWORDS.get(topic, [topic.lower()])
        return any(kw in text for kw in keywords)

    def filter_facts(
        self, facts: List[str], doctor_message: str
    ) -> List[str]:
        """
        Return `facts` with concealed-topic facts removed UNLESS the doctor's
        message contains a trust-respecting opener.

        If the doctor used an opener: all facts pass through (TDP does not
        suppress disclosure once the patient has been invited safely).
        If the doctor did not open: facts matching any concealed topic are
        dropped.
        """
        if not self.concealed_topics:
            return list(facts)
        if self.doctor_used_opener(doctor_message):
            return list(facts)
        out: List[str] = []
        for f in facts:
            if any(
                self._fact_matches_topic(f, topic)
                for topic in self.concealed_topics
            ):
                continue
            out.append(f)
        return out


# ---------- loader -----------------------------------------------------------


def load_trust_persona_from_id(
    persona_id: str,
    personas_dir: Optional[Path] = None,
) -> TrustDisclosurePersona:
    """Load `<persona_id>.yaml` from the trust-personas directory."""
    if personas_dir is None:
        personas_dir = (
            Path(__file__).resolve().parent.parent / "config" / "personas_trust"
        )
    path = personas_dir / f"{persona_id}.yaml"
    return TrustDisclosurePersona.from_yaml(path)


# ---------- public helper for outcome-judge integration ---------------------


def topic_appears_in_text(topic: str, text: str) -> bool:
    """Substring keyword check: does any keyword for `topic` appear in `text`?

    Exposed for the StageJudge to detect whether a concealed topic was
    surfaced during the encounter. Uses the same keyword maps as the
    PatientAgent fact filter, so a single substring list is the audit
    surface for both directions of the disclosure mechanism.

    For unknown topics not in the standard menu, falls back to substring
    matching on the topic string itself.
    """
    if not text or not topic:
        return False
    needle = " " + text.lower() + " "
    keywords = _TOPIC_KEYWORDS.get(topic, [topic.lower()])
    return any(kw in needle for kw in keywords)

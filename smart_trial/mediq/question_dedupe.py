"""Normalize and cap repeated doctor questions within an encounter."""
from __future__ import annotations

import re
from typing import List, Optional, Sequence, Tuple

_RETRIEVAL_PREFIX = re.compile(r"\[RETRIEVAL QUERY:[^\]]*\]\s*", re.IGNORECASE)

SYMPTOMS_FALLBACK = "Could you tell me more about your symptoms?"
TESTS_TREATMENTS_FALLBACK = (
    "Could you tell me more about any tests or treatments you have had for this problem?"
)
CARE_DETAIL_FALLBACK = (
    "Is there any other test result or treatment detail from your recent care "
    "that we have not discussed yet?"
)

FALLBACK_QUESTION_POOL: Tuple[str, ...] = (
    SYMPTOMS_FALLBACK,
    TESTS_TREATMENTS_FALLBACK,
    CARE_DETAIL_FALLBACK,
    "When did your symptoms start?",
    "Have you been started on any antibiotics or other medications for these symptoms?",
    "Was any fluid taken from a swollen joint for testing, and what were the results?",
    "Do you know whether the bacteria from your cultures ferment maltose?",
    "Have you had any recent sexual partners or new partners before these symptoms began?",
    "Have you noticed any rash, eye redness, or sore throat along with these symptoms?",
)

_UNINFORMATIVE_PATTERNS: Tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bi\s+don'?t\s+know\b",
        r"\bi\s+do\s+not\s+know\b",
        r"\bnot\s+(?:totally\s+)?sure\b",
        r"\bcan'?t\s+remember\b",
        r"\bcannot\s+remember\b",
        r"\bno\s+idea\b",
        r"\bwhy\s+does\s+that\s+matter\b",
        r"\bwhy\s+do\s+you\s+need\b",
        r"\bwhy\s+do\s+you\s+want\b",
        r"\bi\s+haven'?t\s+noticed\b",
        r"\bi\s+have\s+not\s+noticed\b",
    )
)


def normalize_question(text: str) -> str:
    """Comparable form for dedupe (strip RAG markers, collapse whitespace, lowercase)."""
    cleaned = _RETRIEVAL_PREFIX.sub("", text or "").strip()
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned.rstrip("?.! ")


def extract_doctor_questions(conversation_history: List[dict]) -> List[str]:
    """Prior doctor lines that contain a patient-directed question."""
    out: List[str] = []
    for msg in conversation_history:
        if msg.get("role") != "assistant":
            continue
        content = (msg.get("content") or "").strip()
        if not content or "?" not in content:
            continue
        if "[DIAGNOSIS]" in content.upper():
            continue
        out.append(content)
    return out


def count_normalized_matches(question: str, prior_questions: List[str]) -> int:
    norm = normalize_question(question)
    if not norm:
        return 0
    return sum(1 for q in prior_questions if normalize_question(q) == norm)


def exceeds_repeat_limit(
    question: str,
    prior_questions: List[str],
    *,
    max_occurrences: int = 2,
) -> bool:
    """True when asking ``question`` would exceed ``max_occurrences`` (default: allow 1 repeat)."""
    if max_occurrences < 1:
        max_occurrences = 1
    return count_normalized_matches(question, prior_questions) >= max_occurrences


def is_uninformative_patient_reply(text: str) -> bool:
    """True when the patient indicates they cannot or will not answer."""
    cleaned = (text or "").strip()
    if not cleaned:
        return True
    return any(p.search(cleaned) for p in _UNINFORMATIVE_PATTERNS)


def patient_rejected_question(question: str, conversation_history: List[dict]) -> bool:
    """True if the last patient reply after this question was uninformative."""
    norm = normalize_question(question)
    if not norm:
        return False
    last_reply: Optional[str] = None
    for i, msg in enumerate(conversation_history):
        if msg.get("role") != "assistant":
            continue
        content = (msg.get("content") or "").strip()
        if "?" not in content or normalize_question(content) != norm:
            continue
        for j in range(i + 1, len(conversation_history)):
            if conversation_history[j].get("role") == "user":
                last_reply = conversation_history[j].get("content") or ""
                break
    if last_reply is None:
        return False
    return is_uninformative_patient_reply(last_reply)


def should_block_question(
    question: str,
    conversation_history: List[dict],
    *,
    max_occurrences: int = 2,
) -> bool:
    """Block when repeat cap hit or patient already gave an uninformative answer."""
    prior = extract_doctor_questions(conversation_history)
    if exceeds_repeat_limit(question, prior, max_occurrences=max_occurrences):
        return True
    if patient_rejected_question(question, conversation_history):
        return True
    return False


def pick_available_question(
    candidates: Sequence[str],
    conversation_history: List[dict],
    *,
    max_occurrences: int = 2,
    start_index: int = 0,
) -> Optional[str]:
    """First candidate that is not blocked by repeat cap or patient rejection."""
    if not candidates:
        return None
    n = len(candidates)
    for offset in range(n):
        q = candidates[(start_index + offset) % n]
        if not should_block_question(q, conversation_history, max_occurrences=max_occurrences):
            return q
    return None


def pick_fallback_question(
    conversation_history: List[dict],
    *,
    max_occurrences: int = 2,
    turn_index: int = 1,
) -> Optional[str]:
    """Rotate through fallback pool; every option must pass repeat and rejection checks."""
    start = max(0, turn_index - 1) % len(FALLBACK_QUESTION_POOL)
    return pick_available_question(
        FALLBACK_QUESTION_POOL,
        conversation_history,
        max_occurrences=max_occurrences,
        start_index=start,
    )


def fact_directed_questions(case: dict) -> List[str]:
    """Simple patient-facing questions derived from case atomic facts."""
    out: List[str] = []
    for raw in case.get("atomic_facts") or []:
        fact = str(raw).strip()
        if not fact:
            continue
        fact = re.sub(r"^\d+\.\s*", "", fact)
        if fact.endswith("?"):
            out.append(fact)
        else:
            out.append(f"Can you tell me whether this applies to you: {fact}?")
    return out


def pick_fact_directed_question(
    case: dict,
    conversation_history: List[dict],
    *,
    max_occurrences: int = 2,
    turn_index: int = 1,
) -> Optional[str]:
    candidates = fact_directed_questions(case)
    if not candidates:
        return None
    start = max(0, turn_index - 1) % len(candidates)
    return pick_available_question(
        candidates,
        conversation_history,
        max_occurrences=max_occurrences,
        start_index=start,
    )


def format_exclude_block(exclude_questions: List[str]) -> str:
    """Prompt appendix listing questions already asked."""
    unique: List[str] = []
    seen = set()
    for q in exclude_questions:
        norm = normalize_question(q)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        display = _RETRIEVAL_PREFIX.sub("", q).strip()
        unique.append(display)
    if not unique:
        return ""
    lines = "\n".join(f"- {item}" for item in unique[-12:])
    return (
        "\n\nQuestions already asked (do NOT repeat or paraphrase into the same question):\n"
        f"{lines}"
    )

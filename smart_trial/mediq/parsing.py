"""Parse MediQ expert LM outputs (choice vs atomic question vs rationale)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ParsedRGResponse:
    reason: Optional[str] = None
    letter_choice: Optional[str] = None
    atomic_question: Optional[str] = None
    abstain: bool = False


def parse_atomic_question(response_text: str) -> Optional[str]:
    questions = []
    for line in response_text.split("\n"):
        if "?" in line:
            questions.append(line.split(":")[-1].strip())
    if not questions:
        return None
    return questions[-1].replace("'", "").replace('"', "").strip()


def parse_choice(response_text: str, options_dict: Dict[str, str]) -> Optional[str]:
    text = response_text.strip()
    if text in ("A", "B", "C", "D"):
        return text
    for response_line in text.split("\n"):
        for _op_letter, op_text in options_dict.items():
            if op_text.lower() in response_line.lower():
                return _op_letter
        for op_letter in options_dict.keys():
            tokens = re.sub(r"[,.;@#()?!'/&:$]+\ *", " ", response_line).split()
            if op_letter in tokens:
                return op_letter
    return None


def _extract_labeled_field(text: str, label: str) -> Optional[str]:
    pattern = rf"(?im)^{re.escape(label)}\s*:?\s*(.+)$"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None


def parse_rg_response(response_text: str, options_dict: Dict[str, str]) -> ParsedRGResponse:
    """Parse implicit_RG / grounded_answer_RG structured output."""
    text = (response_text or "").strip()
    if not text:
        return ParsedRGResponse()

    reason = _extract_labeled_field(text, "REASON")
    answer_line = _extract_labeled_field(text, "ANSWER")
    question_line = _extract_labeled_field(text, "QUESTION")

    if re.search(r"(?im)^ABSTAIN\s*$", text) or (answer_line and answer_line.upper().strip() == "ABSTAIN"):
        return ParsedRGResponse(reason=reason, abstain=True)

    letter: Optional[str] = None
    if answer_line:
        letter = parse_choice(answer_line, options_dict)
    if letter is None:
        letter = parse_choice(text, options_dict)

    atomic: Optional[str] = None
    if question_line and "?" in question_line:
        atomic = question_line
    elif "?" in text and letter is None:
        atomic = parse_atomic_question(text)

    if letter:
        return ParsedRGResponse(reason=reason, letter_choice=letter)
    if atomic:
        return ParsedRGResponse(reason=reason, atomic_question=atomic, abstain=True)
    if "?" in text:
        return ParsedRGResponse(reason=reason, atomic_question=parse_atomic_question(text), abstain=True)
    return ParsedRGResponse(reason=reason, abstain=True)

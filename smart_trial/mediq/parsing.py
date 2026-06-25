"""Parse MediQ expert LM outputs (choice vs atomic question)."""
from __future__ import annotations

import re
from typing import Dict, Optional


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

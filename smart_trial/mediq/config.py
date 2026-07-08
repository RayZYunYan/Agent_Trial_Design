"""MediQ integration settings (loaded from trial_config.yaml ``mediq`` section)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MediQConfig:
    enabled: bool = False
    expert_class: str = "BasicExpert"
    rationale_generation: bool = True
    self_consistency: int = 1
    stage1_suppress_answer: bool = True
    stage2_allow_mcq_finalize: bool = True
    shadow_choice_enabled: bool = False
    max_question_occurrences: int = 2

    @classmethod
    def from_dict(cls, raw: Dict[str, Any] | None) -> "MediQConfig":
        if not raw:
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            expert_class=str(raw.get("expert_class", "BasicExpert")),
            rationale_generation=bool(raw.get("rationale_generation", True)),
            self_consistency=max(1, int(raw.get("self_consistency", 1))),
            stage1_suppress_answer=bool(raw.get("stage1_suppress_answer", True)),
            stage2_allow_mcq_finalize=bool(raw.get("stage2_allow_mcq_finalize", True)),
            shadow_choice_enabled=bool(raw.get("shadow_choice_enabled", False)),
            max_question_occurrences=max(1, int(raw.get("max_question_occurrences", 2))),
        )

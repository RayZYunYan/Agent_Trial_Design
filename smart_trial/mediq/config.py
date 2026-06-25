"""MediQ integration settings (loaded from trial_config.yaml ``mediq`` section)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class MediQConfig:
    enabled: bool = False
    expert_class: str = "BasicExpert"
    rationale_generation: bool = False
    self_consistency: int = 1
    stage1_suppress_answer: bool = True
    stage2_allow_mcq_finalize: bool = True

    @classmethod
    def from_dict(cls, raw: Dict[str, Any] | None) -> "MediQConfig":
        if not raw:
            return cls()
        return cls(
            enabled=bool(raw.get("enabled", False)),
            expert_class=str(raw.get("expert_class", "BasicExpert")),
            rationale_generation=bool(raw.get("rationale_generation", False)),
            self_consistency=max(1, int(raw.get("self_consistency", 1))),
            stage1_suppress_answer=bool(raw.get("stage1_suppress_answer", True)),
            stage2_allow_mcq_finalize=bool(raw.get("stage2_allow_mcq_finalize", True)),
        )

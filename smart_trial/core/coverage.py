"""Fact-coverage early-stop session (incremental LLM judge, monotonic merge)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from smart_trial.core.judge import StageJudge


def parse_coverage_config(trial_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Hot-swappable fact-coverage settings under ``trial.coverage``."""
    rounds = dict(trial_cfg.get("rounds") or {})
    stage1 = dict(rounds.get("stage1") or {})
    stage2 = dict(rounds.get("stage2") or {})
    cov = dict(trial_cfg.get("coverage") or {})
    threshold = float(cov.get("threshold", 0.50))
    return {
        "enabled": bool(cov.get("enabled", True)),
        "check_every": max(1, int(cov.get("check_every", 5))),
        "threshold": threshold,
        "gate_mcq_finalize": bool(cov.get("gate_mcq_finalize", True)),
        "stage1_min_turns": int(cov.get("stage1_min_turns", stage1.get("min_turns", 2))),
        "stage2_min_turns": int(cov.get("stage2_min_turns", stage2.get("min_turns", 2))),
        "summary_subdir": str(cov.get("summary_subdir", "coverage_summaries")),
        "r2_high_threshold": float(
            cov.get("r2_high_threshold", trial_cfg.get("R2_high_confidence_threshold", 0.7))
        ),
    }


def coverage_to_r1_snapshot(coverage: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    """Map coverage state to legacy R1-shaped dict for stage transition / pools."""
    rate = float(coverage.get("coverage_rate", 0.0))
    count = int(coverage.get("coverage_count", 0))
    total = int(coverage.get("total_facts", 0))
    responder = bool(coverage.get("responder", rate >= threshold))
    return {
        "coverage_rate": rate,
        "coverage_count": count,
        "total_facts": total,
        "total": int(round(rate * 10)),
        "responder": responder,
        "covered_facts": list(coverage.get("covered_facts") or []),
        "reasoning": (
            f"fact coverage {count}/{total} ({rate:.0%}); "
            f"responder={responder} (threshold {threshold:.0%})"
        ),
    }


@dataclass
class CoverageSession:
    """Tracks monotonic fact coverage across an encounter."""

    judge: StageJudge
    case: Dict[str, Any]
    threshold: float
    check_every: int
    atomic_facts: List[str] = field(default_factory=list)
    covered_indices: Set[int] = field(default_factory=set)
    covered_facts: List[Dict[str, Any]] = field(default_factory=list)
    last_history_len: int = 0

    def __post_init__(self) -> None:
        raw = self.case.get("atomic_facts") or []
        self.atomic_facts = [str(f).strip() for f in raw if str(f).strip()]

    @property
    def total_facts(self) -> int:
        return len(self.atomic_facts)

    @property
    def coverage_count(self) -> int:
        return len(self.covered_indices)

    @property
    def coverage_rate(self) -> float:
        if not self.atomic_facts:
            return 0.0
        return self.coverage_count / len(self.atomic_facts)

    def meets_threshold(self) -> bool:
        return self.coverage_rate >= self.threshold

    def snapshot(self) -> Dict[str, Any]:
        return {
            "coverage_rate": round(self.coverage_rate, 4),
            "coverage_count": self.coverage_count,
            "total_facts": self.total_facts,
            "responder": self.meets_threshold(),
            "covered_facts": list(self.covered_facts),
            "covered_indices": sorted(self.covered_indices),
        }

    def should_probe(self, turn: int) -> bool:
        return turn > 0 and turn % self.check_every == 0

    def probe_if_due(
        self,
        turn: int,
        conversation_history: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not self.should_probe(turn):
            return None
        new_slice = conversation_history[self.last_history_len :]
        if not new_slice and self.last_history_len > 0:
            self.last_history_len = len(conversation_history)
            return None
        delta = self.judge.probe_fact_coverage_incremental(
            self.case,
            new_conversation=new_slice,
            atomic_facts=self.atomic_facts,
            already_covered=sorted(self.covered_indices),
        )
        self._merge_delta(delta)
        self.last_history_len = len(conversation_history)
        snap = self.snapshot()
        snap["turn"] = turn
        snap["newly_covered_this_probe"] = delta.get("newly_covered") or []
        return snap

    def final_audit(self, conversation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """End-of-encounter pass for human-readable summary (full dialogue)."""
        remaining = [i for i in range(1, self.total_facts + 1) if i not in self.covered_indices]
        if remaining:
            tail = conversation_history[self.last_history_len :]
            if tail:
                delta = self.judge.probe_fact_coverage_incremental(
                    self.case,
                    new_conversation=tail,
                    atomic_facts=self.atomic_facts,
                    already_covered=sorted(self.covered_indices),
                )
                self._merge_delta(delta)
            if remaining:
                full_delta = self.judge.probe_fact_coverage_incremental(
                    self.case,
                    new_conversation=conversation_history,
                    atomic_facts=self.atomic_facts,
                    already_covered=sorted(self.covered_indices),
                    review_mode=True,
                )
                self._merge_delta(full_delta)
        self.last_history_len = len(conversation_history)
        return self.snapshot()

    def _merge_delta(self, delta: Dict[str, Any]) -> None:
        for item in delta.get("newly_covered") or []:
            try:
                idx = int(item.get("index"))
            except (TypeError, ValueError):
                continue
            if idx in self.covered_indices:
                continue
            if idx < 1 or idx > self.total_facts:
                continue
            self.covered_indices.add(idx)
            fact_text = str(item.get("fact") or self.atomic_facts[idx - 1])
            self.covered_facts.append(
                {
                    "index": idx,
                    "fact": fact_text,
                    "patient_evidence": str(item.get("patient_evidence") or ""),
                }
            )


def write_coverage_summary(
    output_dir: Optional[Path],
    case_id: str,
    encounter_id: str,
    coverage: Dict[str, Any],
    *,
    subdir: str = "coverage_summaries",
) -> Optional[Path]:
    if output_dir is None:
        return None
    dest_dir = output_dir / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"{case_id}_{encounter_id}_coverage.txt"
    lines = [
        f"Case: {case_id}",
        f"Encounter: {encounter_id}",
        f"Coverage: {coverage.get('coverage_count', 0)}/{coverage.get('total_facts', 0)} "
        f"({float(coverage.get('coverage_rate', 0)):.1%})",
        f"Responder (>= threshold): {coverage.get('responder')}",
        "",
        "Covered atomic facts:",
    ]
    for item in coverage.get("covered_facts") or []:
        lines.append(f"  [{item.get('index')}] {item.get('fact')}")
        evidence = item.get("patient_evidence")
        if evidence:
            lines.append(f"      Patient evidence: {evidence}")
    uncovered = set(range(1, int(coverage.get("total_facts") or 0) + 1)) - {
        int(x.get("index")) for x in (coverage.get("covered_facts") or []) if x.get("index")
    }
    if uncovered:
        lines.append("")
        lines.append(f"Not covered indices: {sorted(uncovered)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


class TrajectoryLogger:
    """Append one JSON object per encounter to aggregate and/or per-case JSONL.

    When ``aggregate_filename`` is set, rows are written only to that file (eval runs).
    Otherwise each encounter appends to ``{case_id}.jsonl`` (legacy SMART trials).
    """

    def __init__(self, output_dir: str, aggregate_filename: Optional[str] = None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.aggregate_filename = aggregate_filename
        self._current: Dict[str, Any] = {}
        self._turns: List[Dict[str, Any]] = []
        self._stage2_confidences: List[float] = []

    def start_encounter(
        self,
        case: Dict[str, Any],
        seed: int,
        stage1_arm: Optional[str],
        persona: Optional[Dict[str, Any]] = None,
        *,
        run_mode: str = "smart_random",
        path_id: Optional[str] = None,
        forced_a1: Optional[str] = None,
        forced_a2: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._turns = []
        self._stage2_confidences = []
        self._current = {
            "encounter_id": f"enc_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "run_mode": run_mode,
            "path_id": path_id,
            "forced_a1": forced_a1,
            "forced_a2": forced_a2,
            "case_id": case["case_id"],
            "case_category": case.get("case_category", "Other"),
            "chief_complaint": case.get("chief_complaint", ""),
            "ground_truth": case.get("ground_truth_answer", ""),
            "seed": seed,
            "persona": persona or {},
            "stage1_arm": stage1_arm,
            "R1": None,
            "stage2_arm": None,
            "R2": None,
            "outcome": None,
            "total_turns": 0,
            "timestamp_start": datetime.now().isoformat(),
            "trajectory": [],
        }
        if extra:
            self._current.update(extra)

    def log_turn(
        self,
        turn_number: int,
        stage: int,
        arm_id: str,
        doctor_message: str,
        patient_message: str,
        confidence: Optional[float] = None,
    ) -> None:
        turn_record: Dict[str, Any] = {
            "turn": turn_number,
            "stage": stage,
            "arm": arm_id,
            "doctor": doctor_message,
            "patient": patient_message,
        }
        if confidence is not None:
            turn_record["confidence"] = confidence
            self._stage2_confidences.append(confidence)

        self._turns.append(turn_record)
        self._current["total_turns"] = turn_number

    def log_stage_transition(self, stage: int, R_score: Dict[str, Any], next_arm: str, pool_info: Dict[str, Any]) -> None:
        if stage == 1:
            self._current["R1"] = R_score
            self._current["stage2_arm"] = next_arm
            self._current["stage2_pool"] = pool_info.get("pool_used")

    def log_R2(self, R_score: Dict[str, Any]) -> None:
        """R2 is recorded as a measured covariate (no Stage-3 re-randomization)."""
        self._current["R2"] = R_score

    def get_stage2_confidences(self) -> List[float]:
        return list(self._stage2_confidences)

    def finalize(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        self._current["outcome"] = outcome
        self._current["trajectory"] = self._turns
        self._current["timestamp_end"] = datetime.now().isoformat()

        line = json.dumps(self._current, ensure_ascii=False) + "\n"
        if self.aggregate_filename:
            aggregate_path = os.path.join(self.output_dir, self.aggregate_filename)
            with open(aggregate_path, "a", encoding="utf-8") as f:
                f.write(line)
        else:
            output_file = os.path.join(self.output_dir, f"{self._current['case_id']}.jsonl")
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(line)

        return self._current

    @staticmethod
    def load_all_encounters(output_dir: str) -> List[Dict[str, Any]]:
        encounters: List[Dict[str, Any]] = []
        if not os.path.isdir(output_dir):
            return encounters
        for fname in os.listdir(output_dir):
            if fname.endswith(".jsonl"):
                path = os.path.join(output_dir, fname)
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            encounters.append(json.loads(line))
        return encounters

    @staticmethod
    def load_aggregate(aggregate_path: str) -> List[Dict[str, Any]]:
        path = aggregate_path
        if not os.path.isfile(path):
            return []
        out: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class TrajectoryLogger:
    """Append one JSON object per encounter to a JSONL destination.

    Write targets (first match wins):
      1. ``output_file`` — single aggregate file (SMART batch runs)
      2. ``aggregate_filename`` under ``output_dir`` — eval runs
      3. ``{case_id}.jsonl`` under ``output_dir`` — legacy per-case layout
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        *,
        output_file: Optional[str] = None,
        aggregate_filename: Optional[str] = None,
    ):
        self.output_dir = output_dir
        self.output_file = str(output_file) if output_file else None
        self.aggregate_filename = aggregate_filename

        if self.output_file:
            parent = os.path.dirname(self.output_file)
            if parent:
                os.makedirs(parent, exist_ok=True)
        elif output_dir:
            os.makedirs(output_dir, exist_ok=True)

        self._current: Dict[str, Any] = {}
        self._turns: List[Dict[str, Any]] = []
        self._stage2_confidences: List[float] = []

    def _write_path(self, case_id: str) -> str:
        if self.output_file:
            return self.output_file
        if self.aggregate_filename and self.output_dir:
            return os.path.join(self.output_dir, self.aggregate_filename)
        if self.output_dir:
            return os.path.join(self.output_dir, f"{case_id}.jsonl")
        raise ValueError("TrajectoryLogger has no output destination configured")

    def start_encounter(
        self,
        case: Dict[str, Any],
        seed: int,
        stage1_arm: Optional[str],
        persona: Optional[Dict[str, Any]] = None,
        *,
        run_mode: Optional[str] = None,
        path_id: Optional[str] = None,
        forced_a1: Optional[str] = None,
        forced_a2: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._turns = []
        self._stage2_confidences = []
        self._current = {
            "encounter_id": f"enc_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
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
        if run_mode is not None:
            self._current["run_mode"] = run_mode
        if path_id is not None:
            self._current["path_id"] = path_id
        if forced_a1 is not None:
            self._current["forced_a1"] = forced_a1
        if forced_a2 is not None:
            self._current["forced_a2"] = forced_a2
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
        mediq_meta: Optional[Dict[str, Any]] = None,
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
        if mediq_meta:
            turn_record["mediq"] = mediq_meta
            letter = mediq_meta.get("letter_choice")
            if letter:
                turn_record["shadow_letter"] = letter

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
        with open(self._write_path(self._current["case_id"]), "a", encoding="utf-8") as f:
            f.write(line)

        return self._current

    @staticmethod
    def load_logged_case_ids(path: str) -> set[str]:
        """case_id values already present in the log (for skip-on-resume)."""
        ids: set[str] = set()
        for row in TrajectoryLogger.load_all_encounters(path):
            case_id = row.get("case_id")
            if case_id:
                ids.add(case_id)
        return ids

    @staticmethod
    def load_all_encounters(path: str) -> List[Dict[str, Any]]:
        """Load from a single JSONL file or a directory of per-case *.jsonl files."""
        p = Path(path)
        if p.is_file():
            return TrajectoryLogger._read_jsonl_file(p)
        if p.is_dir():
            encounters: List[Dict[str, Any]] = []
            for fname in sorted(os.listdir(p)):
                if fname.endswith(".jsonl"):
                    encounters.extend(TrajectoryLogger._read_jsonl_file(p / fname))
            return encounters
        return []

    @staticmethod
    def load_aggregate(aggregate_path: str) -> List[Dict[str, Any]]:
        return TrajectoryLogger._read_jsonl_file(Path(aggregate_path))

    @staticmethod
    def _read_jsonl_file(path: Path) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if not path.is_file():
            return records
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

import json
import tempfile
from pathlib import Path

from smart_trial.trajectory_log.trajectory_logger import TrajectoryLogger

_CASE = {
    "case_id": "test_case_001",
    "case_category": "Other",
    "chief_complaint": "headache",
    "ground_truth_answer": "migraine",
}


def _minimal_encounter(stage1_arm: str = "A1a") -> dict:
    return {
        "case": _CASE,
        "seed": 1,
        "stage1_arm": stage1_arm,
        "persona": {},
    }


def _finalize_stub(logger: TrajectoryLogger, enc: dict) -> dict:
    logger.start_encounter(enc["case"], enc["seed"], enc["stage1_arm"], persona=enc["persona"])
    logger.log_turn(1, 1, enc["stage1_arm"], "doc", "pat")
    return logger.finalize({"diag_correct": True})


def test_single_file_output_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "encounters.jsonl"
        logger = TrajectoryLogger(output_file=str(path))
        _finalize_stub(logger, _minimal_encounter())

        assert path.is_file()
        rows = TrajectoryLogger.load_all_encounters(str(path))
        assert len(rows) == 1
        assert rows[0]["case_id"] == "test_case_001"
        assert TrajectoryLogger.load_logged_case_ids(str(path)) == {"test_case_001"}


def test_aggregate_filename_eval_mode():
    with tempfile.TemporaryDirectory() as tmp:
        logger = TrajectoryLogger(tmp, aggregate_filename="grid_encounters.jsonl")
        enc = _minimal_encounter("A1b")
        enc["run_mode"] = "smart_grid"
        enc["path_id"] = "A1b_A2a"
        logger.start_encounter(
            enc["case"],
            enc["seed"],
            enc["stage1_arm"],
            persona=enc["persona"],
            run_mode="smart_grid",
            path_id="A1b_A2a",
        )
        logger.finalize({"diag_correct": False})

        agg = Path(tmp) / "grid_encounters.jsonl"
        assert agg.is_file()
        rows = TrajectoryLogger.load_aggregate(str(agg))
        assert len(rows) == 1
        assert rows[0]["run_mode"] == "smart_grid"
        assert rows[0]["path_id"] == "A1b_A2a"


def test_legacy_per_case_directory():
    with tempfile.TemporaryDirectory() as tmp:
        logger = TrajectoryLogger(tmp)
        _finalize_stub(logger, _minimal_encounter())

        case_file = Path(tmp) / "test_case_001.jsonl"
        assert case_file.is_file()
        rows = TrajectoryLogger.load_all_encounters(tmp)
        assert len(rows) == 1


def test_load_all_encounters_reads_directory_of_files():
    with tempfile.TemporaryDirectory() as tmp:
        case_path = Path(tmp) / "a.jsonl"
        row = {"case_id": "a", "stage1_arm": "A1a"}
        case_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

        rows = TrajectoryLogger.load_all_encounters(tmp)
        assert len(rows) == 1
        assert rows[0]["case_id"] == "a"

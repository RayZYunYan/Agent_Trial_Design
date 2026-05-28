import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from smart_trial.core.doctor_agent import DoctorAgent, load_arm_config
from smart_trial.core.judge import StageJudge
from smart_trial.core.patient_agent import PERSONA_AXES, PatientAgent
from smart_trial.core.randomizer import TrialRandomizer
from smart_trial.trajectory_log.trajectory_logger import TrajectoryLogger
from smart_trial.models.model_client import ModelClient


def _sample_persona(rng: random.Random) -> Dict[str, str]:
    return {axis: rng.choice(values) for axis, values in PERSONA_AXES.items()}

SMART_TRIAL_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SMART_TRIAL_ROOT.parent


def resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


class TrialOrchestrator:
    """
    Stage 1: turns 1–4 (after opening + patient acknowledgement).
    Stage 2: turns 5–10.
    Stage 3: turns 11–20 until conclusion or cap.
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            cfg_path = Path(config_path)
            self.config_path = cfg_path if cfg_path.is_absolute() else (PROJECT_ROOT / cfg_path).resolve()
        else:
            self.config_path = (SMART_TRIAL_ROOT / "config" / "trial_config.yaml").resolve()
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.patient_model = self._make_client("patient_simulator")
        self.doctor_model = self._make_client("doctor_agent")
        self.judge_model = self._make_client("judge")

        trial_cfg = self.config.get("trial", {})
        self.judge = StageJudge(
            self.judge_model,
            r1_responder_threshold=int(trial_cfg.get("R1_responder_threshold", 6)),
            r2_high_confidence_threshold=float(trial_cfg.get("R2_high_confidence_threshold", 0.7)),
        )

        log_cfg = self.config.get("logging", {})
        self._output_dir = resolve_path(log_cfg.get("output_dir", "smart_trial/outputs/encounters"))

        self._arms_dir = SMART_TRIAL_ROOT / "config" / "arms"

    def _make_client(self, role: str) -> ModelClient:
        cfg = dict(self.config["models"][role])
        if os.environ.get("SMART_TRIAL_USE_MOCK", "").lower() in ("1", "true", "yes"):
            cfg["provider"] = "mock"
            cfg["model_name"] = "mock"
        return ModelClient(
            provider=cfg["provider"],
            model_name=cfg["model_name"],
            temperature=float(cfg.get("temperature", 0.5)),
        )

    def run_encounter(
        self,
        case: Dict[str, Any],
        seed: Optional[int] = None,
        persona: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if seed is None:
            seed = int(self.config.get("randomization", {}).get("seed", 42))

        rand_cfg = self.config.get("randomization", {})
        randomizer = TrialRandomizer(seed, stratify_by=rand_cfg.get("stratify_by", "case_category"))
        logger = TrajectoryLogger(str(self._output_dir))

        rng = random.Random(seed)
        if persona is None:
            persona = _sample_persona(rng)

        stage1_arm_id = randomizer.assign_stage1_arm(case)
        stage1_arm = load_arm_config(stage1_arm_id, self._arms_dir)

        print(f"\n{'=' * 60}")
        print(f"Case: {case['case_id']} | {case.get('case_category', '')}")
        print(f"Chief Complaint: {case.get('chief_complaint', '')}")
        print(f"Stage 1 Arm: {stage1_arm_id} ({stage1_arm.get('name', '')})")
        print(f"Persona: {persona}")
        print(f"{'=' * 60}\n")

        doctor = DoctorAgent(self.doctor_model, stage1_arm)
        patient = PatientAgent(self.patient_model, case, persona=persona)

        logger.start_encounter(case, seed, stage1_arm_id, persona=persona)

        initial_msg = doctor.get_initial_message(case)
        print(f"[Opening]\nDoctor: {initial_msg}\n")
        last_patient = patient.respond(initial_msg)

        print(f"--- STAGE 1 ({stage1_arm.get('name', '')}) ---")
        for turn in range(1, 5):
            doctor_msg, confidence = doctor.respond(last_patient)
            print(f"[Turn {turn}]")
            print(f"Doctor: {doctor_msg}")
            last_patient = patient.respond(doctor_msg)
            print(f"Patient: {last_patient}\n")

            logger.log_turn(turn, 1, stage1_arm_id, doctor_msg, last_patient, confidence)

        print("--- Computing R1 ---")
        R1 = self.judge.compute_R1(doctor.conversation_history, case)
        print(f"R1 Score: {R1.get('total')}/10 | Responder: {R1.get('responder')}")

        stage2_assignment = randomizer.assign_stage2_arm(case, R1)
        stage2_arm_id = stage2_assignment["arm"]
        stage2_arm = load_arm_config(stage2_arm_id, self._arms_dir)

        print(
            f"Stage 2 Arm: {stage2_arm_id} ({stage2_arm.get('name', '')}) "
            f"[Pool: {stage2_assignment['pool_used']}]\n"
        )

        logger.log_stage_transition(1, R1, stage2_arm_id, stage2_assignment)

        doctor.switch_arm(stage2_arm)
        print(f"--- STAGE 2 ({stage2_arm.get('name', '')}) ---")

        stage2_hist_start = len(doctor.conversation_history)
        for turn in range(5, 11):
            doctor_msg, confidence = doctor.respond(last_patient)
            conf_str = f" [conf={confidence:.2f}]" if confidence is not None else ""
            print(f"[Turn {turn}]{conf_str}")
            print(f"Doctor: {doctor_msg}")
            last_patient = patient.respond(doctor_msg)
            print(f"Patient: {last_patient}\n")

            logger.log_turn(turn, 2, stage2_arm_id, doctor_msg, last_patient, confidence)

        print("--- Computing R2 ---")
        stage2_confidences = logger.get_stage2_confidences()
        thr = float(self.config.get("trial", {}).get("R2_high_confidence_threshold", 0.7))
        stage2_slice = doctor.conversation_history[stage2_hist_start:]
        R2 = self.judge.compute_R2(
            doctor.conversation_history,
            stage2_confidences,
            high_threshold=thr,
            stage2_conversation_slice=stage2_slice,
            case=case,
        )
        print(
            f"R2 Confidence: {R2['confidence_level']} (final={R2['final_confidence']:.2f}, "
            f"source={R2.get('r2_source', '')})"
        )

        stage3_assignment = randomizer.assign_stage3_arm(case, R2)
        stage3_arm_id = stage3_assignment["arm"]
        stage3_arm = load_arm_config(stage3_arm_id, self._arms_dir)

        print(
            f"Stage 3 Arm: {stage3_arm_id} ({stage3_arm.get('name', '')}) "
            f"[Pool: {stage3_assignment['pool_used']}]\n"
        )

        logger.log_stage_transition(2, R2, stage3_arm_id, stage3_assignment)

        doctor.switch_arm(stage3_arm)
        print(f"--- STAGE 3 ({stage3_arm.get('name', '')}) ---")

        turn = 11
        max_turn = int(self.config.get("trial", {}).get("max_turns", 20))
        while not doctor.has_concluded() and turn <= max_turn:
            doctor_msg, confidence = doctor.respond(last_patient)
            print(f"[Turn {turn}]")
            print(f"Doctor: {doctor_msg}")
            last_patient = patient.respond(doctor_msg)
            print(f"Patient: {last_patient}\n")

            logger.log_turn(turn, 3, stage3_arm_id, doctor_msg, last_patient, confidence)
            turn += 1
            if doctor.has_concluded():
                break

        print("--- Evaluating Outcome ---")
        final_diag = doctor.get_final_diagnosis()
        outcome = self.judge.evaluate_outcome(
            final_diagnosis=final_diag or "",
            case=case,
            conversation_history=doctor.conversation_history,
            R2=R2,
        )

        trajectory = logger.finalize(outcome)

        print(f"Diagnosis Correct: {outcome.get('diag_correct')}")
        print(f"Red Flag Miss: {outcome.get('red_flag_miss')}")
        print(f"Dangerous Advice: {outcome.get('dangerous_advice')}")
        print(f"Turns Used: {trajectory.get('total_turns', 0)}")

        print(f"\n{'=' * 60}")
        print(
            f"Trajectory: {trajectory['stage1_arm']} -> "
            f"{trajectory['stage2_arm']} -> {trajectory['stage3_arm']}"
        )
        print(f"Saved to: {self._output_dir / (case['case_id'] + '.jsonl')}")
        print(f"{'=' * 60}\n")

        return trajectory

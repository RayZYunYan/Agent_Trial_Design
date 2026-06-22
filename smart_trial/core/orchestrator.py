import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from smart_trial.core.doctor_agent import DoctorAgent, load_arm_config
from smart_trial.core.judge import StageJudge
from smart_trial.core.patient_agent import PatientAgent
from smart_trial.core.persona import (
    PatientPersona,
    load_persona_from_id,
    select_default_persona_id,
)
from smart_trial.core.randomizer import TrialRandomizer
from smart_trial.trajectory_log.trajectory_logger import TrajectoryLogger
from smart_trial.models.model_client import ModelClient
from smart_trial.rag.retriever import BM25Retriever, ContrieverRetriever, HybridRetriever, BaseRetriever



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
    Stage 2: turns 5–10; the doctor delivers the final [DIAGNOSIS] within this
    stage (early once confident, forced on the last turn).
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
        output_file = log_cfg.get("output_file")
        self._output_file = resolve_path(output_file) if output_file else None
        output_dir = log_cfg.get("output_dir")
        self._output_dir = resolve_path(output_dir) if output_dir else None
        self._aggregate_filename = log_cfg.get("aggregate_filename")

        self._arms_dir = SMART_TRIAL_ROOT / "config" / "arms"
        self._personas_dir = SMART_TRIAL_ROOT / "config" / "personas"

        rag_cfg = self.config.get("rag", {})
        self._rag_enabled = bool(rag_cfg.get("enabled", False))
        self._rag_k = int(rag_cfg.get("k", 3))
        self._rag_index_cache = str(
            resolve_path(rag_cfg.get("index_cache", "smart_trial/data/bm25_index.pkl"))
        )
        self._retriever: Optional[BaseRetriever] = None
        if self._rag_enabled:
            self._retriever = self._init_retriever(rag_cfg)

    # ------------------------------------------------------------------
    # Retriever factory
    # ------------------------------------------------------------------
    def _init_retriever(self, rag_cfg: dict) -> Optional[BaseRetriever]:
        mode = str(rag_cfg.get("retriever_mode", "bm25")).lower()
        bm25_cache = str(resolve_path(rag_cfg.get("index_cache", "smart_trial/data/bm25_index.pkl")))
        contriever_cache = str(resolve_path(
            rag_cfg.get("contriever_cache", "smart_trial/data/contriever_embeddings.npy")
        ))
        contriever_model = rag_cfg.get("contriever_model", "facebook/contriever")
        rrf_k = int(rag_cfg.get("rrf_k", 60))

        try:
            if mode == "bm25":
                print("[RAG] Retriever mode: BM25")
                return BM25Retriever.load(cache_path=bm25_cache)

            if mode == "contriever":
                print("[RAG] Retriever mode: Contriever")
                bm25 = BM25Retriever.load(cache_path=bm25_cache)
                return ContrieverRetriever.load(
                    passages=bm25._passages,
                    cache_path=contriever_cache,
                    model_name=contriever_model,
                )

            if mode == "hybrid":
                print("[RAG] Retriever mode: Hybrid (BM25 + Contriever RRF)")
                bm25 = BM25Retriever.load(cache_path=bm25_cache)
                contriever = ContrieverRetriever.load(
                    passages=bm25._passages,
                    cache_path=contriever_cache,
                    model_name=contriever_model,
                )
                return HybridRetriever(bm25=bm25, contriever=contriever, rrf_k=rrf_k)

            raise ValueError(f"Unknown rag.retriever_mode: {mode!r}. Choose 'bm25', 'contriever', or 'hybrid'.")
        except Exception as exc:
            print(f"[RAG] Warning: could not load retriever ({exc}); retrieval disabled.")
            self._rag_enabled = False
            return None

    # ------------------------------------------------------------------
    # RAG helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_retrieval_query(text: str) -> Optional[str]:
        """Extract query string from [RETRIEVAL QUERY: <q>] in doctor message."""
        import re
        match = re.search(r"\[RETRIEVAL QUERY:\s*(.+?)\]", text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _strip_internal_markers(text: str) -> str:
        """Remove [RETRIEVAL QUERY: ...] lines before sending doctor message to patient."""
        import re
        cleaned = re.sub(r"\[RETRIEVAL QUERY:[^\]]*\]\s*", "", text, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _format_retrieval_result(passages: List[str]) -> str:
        """Format top-k passages into a block injected into doctor system prompt."""
        body = "\n\n".join(passages)
        return f"[RETRIEVAL RESULT]\n{body}\n[/RETRIEVAL RESULT]"

    def _resolve_persona(
        self, case: Dict[str, Any], rng: random.Random
    ) -> Optional[PatientPersona]:
        cfg = self.config.get("persona") or {}
        mode = str(cfg.get("mode", "random")).lower()
        if mode == "off":
            return None
        if mode == "fixed":
            return load_persona_from_id(str(cfg.get("fixed_id", "default")), self._personas_dir)
        if mode == "random":
            case_rng = random.Random(rng.randint(0, 2**32) ^ hash(case.get("case_id", "")))
            return PatientPersona.sample(case_rng)
        if mode == "demographic":
            return load_persona_from_id(select_default_persona_id(case), self._personas_dir)
        if mode == "per_case":
            pid = case.get("persona_id") or select_default_persona_id(case)
            return load_persona_from_id(str(pid), self._personas_dir)
        raise ValueError(f"Unknown persona.mode: {mode!r}")

    @property
    def output_path(self) -> Path:
        """Primary JSONL destination for logging and resume checks."""
        if self._output_file is not None:
            return self._output_file
        if self._aggregate_filename and self._output_dir is not None:
            return self._output_dir / self._aggregate_filename
        if self._output_dir is not None:
            return self._output_dir
        return resolve_path("smart_trial/outputs/encounters.jsonl")

    def _make_logger(self) -> TrajectoryLogger:
        return TrajectoryLogger(
            str(self._output_dir) if self._output_dir is not None else None,
            output_file=str(self._output_file) if self._output_file is not None else None,
            aggregate_filename=self._aggregate_filename,
        )

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
        persona: Optional[PatientPersona] = None,
    ) -> Dict[str, Any]:
        if seed is None:
            seed = int(self.config.get("randomization", {}).get("seed", 42))

        rand_cfg = self.config.get("randomization", {})
        randomizer = TrialRandomizer(seed, stratify_by=rand_cfg.get("stratify_by", "case_category"))
        logger = self._make_logger()

        rng = random.Random(seed)
        if persona is None:
            persona = self._resolve_persona(case, rng)

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

        logger.start_encounter(case, seed, stage1_arm_id, persona=persona.to_dict() if persona else None)

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

        arm_retrieval_enabled = (
            self._rag_enabled
            and bool((stage2_arm.get("tool_access") or {}).get("retrieval", False))
        )

        stage2_hist_start = len(doctor.conversation_history)
        stage2_turns = int(self.config.get("trial", {}).get("stage2_turns", 6))
        last_stage2_turn = 4 + stage2_turns
        pending_retrieval_context: Optional[str] = None
        for turn in range(5, last_stage2_turn + 1):
            # The model cannot count turns itself, so the orchestrator forces the
            # conclusion on the last Stage-2 turn (identical across arms).
            force_conclude = turn == last_stage2_turn
            doctor_msg, confidence = doctor.respond(
                last_patient,
                force_conclude=force_conclude,
                retrieval_context=pending_retrieval_context,
            )
            pending_retrieval_context = None

            if arm_retrieval_enabled and self._retriever is not None:
                query = self._parse_retrieval_query(doctor_msg)
                if query:
                    passages = self._retriever.retrieve(query, k=self._rag_k)
                    pending_retrieval_context = self._format_retrieval_result(passages)
                    print(f"[RAG] Query: {query!r} → {len(passages)} passages retrieved")

            conf_str = f" [conf={confidence:.2f}]" if confidence is not None else ""
            print(f"[Turn {turn}]{conf_str}")
            print(f"Doctor: {doctor_msg}")

            if doctor.has_concluded():
                logger.log_turn(turn, 2, stage2_arm_id, doctor_msg, "", confidence)
                print()
                break

            patient_facing_msg = self._strip_internal_markers(doctor_msg)
            last_patient = patient.respond(patient_facing_msg)
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
            f"source={R2.get('r2_source', '')})\n"
        )

        # R2 is kept as a measured covariate for analysis; it no longer drives
        # any re-randomization (Stage 3 was removed from the design).
        logger.log_R2(R2)

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
            f"Trajectory: {trajectory['stage1_arm']} -> {trajectory['stage2_arm']}"
        )
        print(f"Saved to: {self.output_path}")
        print(f"{'=' * 60}\n")

        return trajectory

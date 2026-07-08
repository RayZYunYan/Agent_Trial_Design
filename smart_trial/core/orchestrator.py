import hashlib
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from smart_trial.core.coverage import (
    CoverageSession,
    coverage_to_r1_snapshot,
    parse_coverage_config,
    write_coverage_summary,
)
from smart_trial.core.doctor_agent import DoctorAgent, load_arm_config
from smart_trial.mediq import MediQConfig
from smart_trial.core.judge import StageJudge
from smart_trial.core.patient_agent import PatientAgent
from smart_trial.core.persona import (
    PatientPersona,
    load_persona_from_id,
    select_default_persona_id,
)
from smart_trial.core.randomizer import TrialRandomizer
from smart_trial.eval.adaptive_policy import AdaptivePolicyAssigner
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


def _stable_persona_seed(global_seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{global_seed}:{case_id}:persona".encode()).hexdigest()
    return int(digest[:16], 16)


def _parse_rounds_config(trial_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve fixed vs adaptive turn limits from trial.rounds."""
    rounds = dict(trial_cfg.get("rounds") or {})
    mode = str(rounds.get("mode", "fixed")).lower()
    stage1 = dict(rounds.get("stage1") or {})
    stage2 = dict(rounds.get("stage2") or {})
    red_flags_min = stage1.get("red_flags_min")
    if red_flags_min is not None:
        red_flags_min = int(red_flags_min)
    return {
        "mode": mode,
        "adaptive": mode == "adaptive",
        "stage1_max": int(trial_cfg.get("stage1_turns", 4)),
        "stage2_max": int(trial_cfg.get("stage2_turns", 6)),
        "stage1_min_turns": int(stage1.get("min_turns", 2)),
        "stage1_check_every": max(1, int(stage1.get("check_every", 1))),
        "stage1_red_flags_min": red_flags_min,
        "stage2_min_turns": int(stage2.get("min_turns", 2)),
    }


def _coverage_allows_early_stop(
    coverage: CoverageSession,
    *,
    min_turns_met: bool,
    threshold: float,
) -> bool:
    if not min_turns_met:
        return False
    return coverage.coverage_rate >= threshold


class TrialOrchestrator:
    """
    Run modes (config ``run.mode``):
      - smart_random: default SMART trial with random arm assignment
      - smart_grid: SMART with forced (A1, A2); ignores Stage-2 pool restrictions
      - smart_adaptive_loop: closed-loop π̂ with biased-random assignment (Phase 2)
      - baseline: single-phase visit, no strategy arms, max_turns dialogue
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

    def _get_run_mode(self, override: Optional[str] = None) -> str:
        if override:
            return override
        return str((self.config.get("run") or {}).get("mode", "smart_random"))

    @property
    def output_path(self) -> Path:
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
        import re
        match = re.search(r"\[RETRIEVAL QUERY:\s*(.+?)\]", text, re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _strip_internal_markers(text: str) -> str:
        import re
        cleaned = re.sub(r"\[RETRIEVAL QUERY:[^\]]*\]\s*", "", text, flags=re.IGNORECASE)
        return cleaned.strip()

    @staticmethod
    def _format_retrieval_result(passages: List[str]) -> str:
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
        if mode == "per_case_seed":
            case_rng = random.Random(_stable_persona_seed(rng.randint(0, 2**31), case["case_id"]))
            return PatientPersona.sample(case_rng)
        raise ValueError(f"Unknown persona.mode: {mode!r}")

    def _persona_for_case(
        self,
        case: Dict[str, Any],
        seed: int,
        persona: Optional[PatientPersona],
    ) -> Optional[PatientPersona]:
        eval_cfg = self.config.get("eval") or {}
        if persona is not None:
            return persona
        if eval_cfg.get("fix_persona_per_case", False):
            case_rng = random.Random(_stable_persona_seed(seed, case["case_id"]))
            return PatientPersona.sample(case_rng)
        return self._resolve_persona(case, random.Random(seed))

    def _parse_coverage_config(self) -> Dict[str, Any]:
        return parse_coverage_config(self.config.get("trial") or {})

    def _parse_rounds_config(self) -> Dict[str, Any]:
        return _parse_rounds_config(self.config.get("trial") or {})

    def _finalize_coverage(
        self,
        logger: TrajectoryLogger,
        coverage: CoverageSession,
        doctor: DoctorAgent,
        cov_cfg: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Return flushed coverage snapshot for R2/gating and summary output."""
        if cov_cfg.get("enabled", True):
            coverage.probe_pending(doctor.conversation_history)
        snap = coverage_to_r1_snapshot(coverage.snapshot(), cov_cfg["threshold"])
        logger._current["final_coverage_rate"] = snap["coverage_rate"]
        logger._current["final_coverage_count"] = snap["coverage_count"]
        logger._current["final_coverage_total"] = snap["total_facts"]
        logger._current["covered_facts"] = snap.get("covered_facts") or []
        logger._current["incremental_coverage_rate"] = snap["coverage_rate"]
        logger._current["incremental_coverage_count"] = snap["coverage_count"]
        enc_id = logger._current.get("encounter_id", "enc")
        if self._output_dir is not None:
            out_dir = Path(self._output_dir)
        elif self._output_file is not None:
            out_dir = Path(self._output_file).parent
        else:
            out_dir = None
        summary_path = write_coverage_summary(
            out_dir,
            logger._current.get("case_id", "case"),
            enc_id,
            snap,
            subdir=cov_cfg.get("summary_subdir", "coverage_summaries"),
        )
        if summary_path:
            logger._current["coverage_summary_file"] = str(summary_path)
        return snap

    def _doctor_allow_finalize(
        self,
        coverage: CoverageSession,
        cov_cfg: Dict[str, Any],
        *,
        stage_turn_idx: int,
        stage_min_turns: int,
        force_conclude: bool,
    ) -> bool:
        if force_conclude:
            return True
        if not cov_cfg.get("enabled", True):
            return True
        if not cov_cfg.get("gate_mcq_finalize", True):
            return True
        if stage_turn_idx < stage_min_turns:
            return False
        return coverage.meets_threshold()

    def _mediq_config(self) -> MediQConfig:
        return MediQConfig.from_dict(self.config.get("mediq"))

    def _make_doctor(
        self,
        arm_config: Dict[str, Any],
        case: Dict[str, Any],
    ) -> DoctorAgent:
        return DoctorAgent(
            self.doctor_model,
            arm_config,
            case=case,
            mediq_config=self._mediq_config(),
        )

    def _attach_mediq_encounter_fields(
        self,
        logger: TrajectoryLogger,
        doctor: DoctorAgent,
    ) -> None:
        if not doctor.mediq_enabled:
            return
        logger._current["mediq"] = {
            "enabled": True,
            "expert_class": doctor.mediq_config.expert_class,
            "intermediate_choices": doctor.get_intermediate_choices(),
            "shadow_choices": doctor.get_shadow_choices(),
            "final_letter_choice": doctor.get_final_letter_choice(),
            "final_rationale": doctor.get_final_rationale(),
        }

    def _evaluate_encounter_outcome(
        self,
        *,
        doctor: DoctorAgent,
        case: Dict[str, Any],
        final_diag: Optional[str],
        R2: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self.judge.evaluate_outcome(
            final_diagnosis=final_diag or "",
            case=case,
            conversation_history=doctor.conversation_history,
            R2=R2,
            mcq_letter_choice=doctor.get_final_letter_choice() if doctor.mediq_enabled else None,
        )

    def _make_client(self, role: str) -> ModelClient:
        cfg = dict(self.config["models"][role])
        use_mock_all = os.environ.get("SMART_TRIAL_USE_MOCK", "").lower() in ("1", "true", "yes")
        use_mock_judge = os.environ.get("SMART_TRIAL_MOCK_JUDGE", "").lower() in ("1", "true", "yes")
        if use_mock_all or (role == "judge" and use_mock_judge):
            cfg["provider"] = "mock"
            cfg["model_name"] = "mock"
        return ModelClient(
            provider=cfg["provider"],
            model_name=cfg["model_name"],
            temperature=float(cfg.get("temperature", 0.5)),
        )

    @staticmethod
    def _neutral_r2() -> Dict[str, Any]:
        return {
            "final_confidence": 0.5,
            "avg_confidence": 0.5,
            "confidence_level": "low",
            "confidence_scores": [],
            "R2_category": "low-confidence",
            "r2_source": "baseline",
        }

    def run_encounter(
        self,
        case: Dict[str, Any],
        seed: Optional[int] = None,
        persona: Optional[PatientPersona] = None,
        *,
        run_mode: Optional[str] = None,
        forced_a1: Optional[str] = None,
        forced_a2: Optional[str] = None,
        path_id: Optional[str] = None,
        policy_assigner: Optional[AdaptivePolicyAssigner] = None,
    ) -> Dict[str, Any]:
        if seed is None:
            seed = int(self.config.get("randomization", {}).get("seed", 42))

        mode = self._get_run_mode(run_mode)
        if mode == "baseline":
            return self._run_baseline(case, seed=seed, persona=persona)
        return self._run_smart(
            case,
            seed=seed,
            persona=persona,
            run_mode=mode,
            forced_a1=forced_a1,
            forced_a2=forced_a2,
            path_id=path_id,
            policy_assigner=policy_assigner,
        )

    def _run_baseline(
        self,
        case: Dict[str, Any],
        *,
        seed: int,
        persona: Optional[PatientPersona],
    ) -> Dict[str, Any]:
        run_cfg = self.config.get("run") or {}
        max_turns = int(run_cfg.get("baseline_max_turns", 11))
        logger = self._make_logger()
        persona = self._persona_for_case(case, seed, persona)

        print(f"\n{'=' * 60}")
        print(f"Case: {case['case_id']} | {case.get('case_category', '')} | mode=baseline")
        print(f"Chief Complaint: {case.get('chief_complaint', '')}")
        print(f"Persona: {persona}")
        print(f"{'=' * 60}\n")

        doctor = self._make_doctor(dict(DoctorAgent.BASELINE_ARM_CONFIG), case)
        patient = PatientAgent(self.patient_model, case, persona=persona)

        cov_cfg = self._parse_coverage_config()
        coverage = CoverageSession(
            self.judge,
            case,
            threshold=cov_cfg["threshold"],
            check_every=cov_cfg["check_every"],
        )
        baseline_min = cov_cfg["stage2_min_turns"]

        logger.start_encounter(
            case,
            seed,
            stage1_arm=None,
            persona=persona.to_dict() if persona else None,
            run_mode="baseline",
        )

        initial_msg = doctor.get_initial_message(case)
        print(f"[Opening]\nDoctor: {initial_msg}\n")
        last_patient = patient.respond(initial_msg)
        if cov_cfg["enabled"]:
            coverage.seed_context_coverage(doctor.conversation_history)

        print("--- BASELINE (no strategy arms) ---")
        for turn in range(1, max_turns + 1):
            force_conclude = turn == max_turns
            allow = self._doctor_allow_finalize(
                coverage,
                cov_cfg,
                stage_turn_idx=turn,
                stage_min_turns=baseline_min,
                force_conclude=force_conclude,
            )
            doctor_msg = doctor.respond(
                last_patient,
                force_conclude=force_conclude,
                allow_finalize=allow,
            )
            print(f"[Turn {turn}]")
            print(f"Doctor: {doctor_msg}")

            if doctor.has_concluded():
                logger.log_turn(
                    turn, 0, "baseline", doctor_msg, "",
                    mediq_meta=doctor.get_last_mediq_meta(),
                )
                print()
                break

            last_patient = patient.respond(doctor_msg)
            print(f"Patient: {last_patient}\n")
            logger.log_turn(
                turn, 0, "baseline", doctor_msg, last_patient,
                mediq_meta=doctor.get_last_mediq_meta(),
            )
            if cov_cfg["enabled"]:
                probe = coverage.probe_if_due(turn, doctor.conversation_history)
                if probe:
                    print(
                        f"[Coverage] {probe['coverage_count']}/{probe['total_facts']} "
                        f"({probe['coverage_rate']:.0%})\n"
                    )

        if cov_cfg["enabled"]:
            coverage.probe_pending(doctor.conversation_history)

        cov_snap = self._finalize_coverage(logger, coverage, doctor, cov_cfg)
        r2 = self.judge.compute_R2_from_coverage(
            cov_snap["coverage_rate"],
            high_threshold=cov_cfg["r2_high_threshold"],
            coverage_count=cov_snap["coverage_count"],
            total_facts=cov_snap["total_facts"],
        )
        logger.log_R2(r2)

        print("--- Evaluating Outcome ---")
        final_diag = doctor.get_final_diagnosis()
        self._attach_mediq_encounter_fields(logger, doctor)
        outcome = self._evaluate_encounter_outcome(
            doctor=doctor, case=case, final_diag=final_diag, R2=r2,
        )

        trajectory = logger.finalize(outcome)
        self._print_outcome(trajectory, label="baseline")
        return trajectory

    def _run_smart(
        self,
        case: Dict[str, Any],
        *,
        seed: int,
        persona: Optional[PatientPersona],
        run_mode: str,
        forced_a1: Optional[str],
        forced_a2: Optional[str],
        path_id: Optional[str],
        policy_assigner: Optional[AdaptivePolicyAssigner] = None,
    ) -> Dict[str, Any]:
        rand_cfg = self.config.get("randomization", {})
        randomizer = TrialRandomizer(seed, stratify_by=rand_cfg.get("stratify_by", "case_category"))
        logger = self._make_logger()
        persona = self._persona_for_case(case, seed, persona)
        persona_dict = persona.to_dict() if persona else {}

        stage1_propensity: Optional[float] = None
        stage1_q: Optional[Dict[str, float]] = None
        if run_mode == "smart_adaptive_loop" and policy_assigner is not None and not forced_a1:
            stage1_arm_id, stage1_propensity, stage1_q = policy_assigner.assign_stage1(
                case, persona_dict, seed
            )
        else:
            stage1_arm_id = forced_a1 or randomizer.assign_stage1_arm(case)
        stage1_arm = load_arm_config(stage1_arm_id, self._arms_dir)
        resolved_path_id = path_id or (
            f"{forced_a1}_{forced_a2}" if forced_a1 and forced_a2 else None
        )

        print(f"\n{'=' * 60}")
        print(f"Case: {case['case_id']} | {case.get('case_category', '')} | mode={run_mode}")
        print(f"Chief Complaint: {case.get('chief_complaint', '')}")
        print(f"Stage 1 Arm: {stage1_arm_id} ({stage1_arm.get('name', '')})")
        if resolved_path_id:
            print(f"Path: {resolved_path_id}")
        if run_mode == "smart_adaptive_loop" and policy_assigner is not None:
            print(f"Policy refit generation: {policy_assigner.refit_generation}")
        print(f"Persona: {persona}")
        print(f"{'=' * 60}\n")

        doctor = self._make_doctor(stage1_arm, case)
        patient = PatientAgent(self.patient_model, case, persona=persona)

        encounter_extra: Optional[Dict[str, Any]] = None
        if run_mode == "smart_adaptive_loop":
            encounter_extra = {
                "assignment_mode": "policy_biased_random" if policy_assigner else "uniform",
                "refit_generation": policy_assigner.refit_generation if policy_assigner else None,
                "stage1_propensity": stage1_propensity,
                "stage1_q_values": stage1_q,
            }

        rounds_cfg = self._parse_rounds_config()
        cov_cfg = self._parse_coverage_config()
        coverage = CoverageSession(
            self.judge,
            case,
            threshold=cov_cfg["threshold"],
            check_every=cov_cfg["check_every"],
        )
        stage1_max = rounds_cfg["stage1_max"]
        stage2_max = rounds_cfg["stage2_max"]
        adaptive_rounds = rounds_cfg["adaptive"]
        stage1_min = cov_cfg["stage1_min_turns"]
        stage2_min = cov_cfg["stage2_min_turns"]
        if encounter_extra is None:
            encounter_extra = {}
        encounter_extra["rounds_mode"] = rounds_cfg["mode"]
        encounter_extra["coverage_threshold"] = cov_cfg["threshold"]

        logger.start_encounter(
            case,
            seed,
            stage1_arm_id,
            persona=persona_dict or None,
            run_mode=run_mode,
            path_id=resolved_path_id,
            forced_a1=forced_a1,
            forced_a2=forced_a2,
            extra=encounter_extra,
        )

        initial_msg = doctor.get_initial_message(case)
        print(f"[Opening]\nDoctor: {initial_msg}\n")
        last_patient = patient.respond(initial_msg)
        if cov_cfg["enabled"]:
            coverage.seed_context_coverage(doctor.conversation_history)

        print(f"--- STAGE 1 ({stage1_arm.get('name', '')}) ---")
        if adaptive_rounds:
            print(
                f"(rounds: adaptive, max={stage1_max}, min={stage1_min}, "
                f"coverage>={cov_cfg['threshold']:.0%}, check_every={cov_cfg['check_every']})"
            )

        R1: Optional[Dict[str, Any]] = None
        stage1_early_stop: Optional[str] = None
        stage1_turns_used = 0
        for turn in range(1, stage1_max + 1):
            stage1_turns_used = turn
            allow = self._doctor_allow_finalize(
                coverage,
                cov_cfg,
                stage_turn_idx=turn,
                stage_min_turns=stage1_min,
                force_conclude=False,
            )
            doctor_msg = doctor.respond(last_patient, allow_finalize=allow)
            print(f"[Turn {turn}]")
            print(f"Doctor: {doctor_msg}")
            last_patient = patient.respond(doctor_msg)
            print(f"Patient: {last_patient}\n")
            logger.log_turn(
                turn, 1, stage1_arm_id, doctor_msg, last_patient,
                mediq_meta=doctor.get_last_mediq_meta(),
            )

            if cov_cfg["enabled"]:
                probe = coverage.probe_if_due(turn, doctor.conversation_history)
                if probe:
                    print(
                        f"[Coverage] {probe['coverage_count']}/{probe['total_facts']} "
                        f"({probe['coverage_rate']:.0%})\n"
                    )

            if (
                adaptive_rounds
                and turn >= stage1_min
                and cov_cfg["enabled"]
            ):
                coverage.probe_pending(doctor.conversation_history, turn=turn)

            if (
                adaptive_rounds
                and turn >= stage1_min
                and _coverage_allows_early_stop(
                    coverage,
                    min_turns_met=True,
                    threshold=cov_cfg["threshold"],
                )
            ):
                R1 = coverage_to_r1_snapshot(coverage.snapshot(), cov_cfg["threshold"])
                stage1_early_stop = "fact_coverage"
                print(
                    f"[Coverage] {coverage.coverage_count}/{coverage.total_facts} "
                    f"({coverage.coverage_rate:.0%})\n"
                )
                print(
                    f"[Adaptive] coverage {coverage.coverage_rate:.0%} — "
                    f"early Stage 1 stop at turn {turn}\n"
                )
                break

        if R1 is None:
            print("--- Stage 1 coverage (R1) ---")
            if cov_cfg["enabled"]:
                coverage.probe_pending(doctor.conversation_history, turn=stage1_turns_used)
            R1 = coverage_to_r1_snapshot(coverage.snapshot(), cov_cfg["threshold"])
        else:
            print("--- R1 (coverage checkpoint) ---")
        print(
            f"Coverage: {R1.get('coverage_count')}/{R1.get('total_facts')} "
            f"({R1.get('coverage_rate', 0):.0%}) | Responder: {R1.get('responder')}"
        )

        logger._current["stage1_turns_used"] = stage1_turns_used
        logger._current["stage1_turns_max"] = stage1_max
        if stage1_early_stop:
            logger._current["stage1_early_stop"] = stage1_early_stop

        if forced_a2:
            pool_key = "responder" if R1.get("responder") else "non-responder"
            stage2_arm_id = forced_a2
            stage2_assignment = {
                "arm": forced_a2,
                "pool_used": pool_key,
                "pool": TrialRandomizer.STAGE2_POOLS[pool_key],
                "R1_total": int(float(R1.get("coverage_rate", 0)) * 10),
                "forced": True,
            }
        elif run_mode == "smart_adaptive_loop" and policy_assigner is not None:
            stage2_assignment = policy_assigner.assign_stage2(
                case,
                persona_dict,
                stage1_arm_id,
                R1,
                stage1_turns_used,
                seed,
            )
            stage2_arm_id = stage2_assignment["arm"]
            logger._current["stage2_propensity"] = stage2_assignment.get("propensity")
            logger._current["stage2_q_values"] = stage2_assignment.get("q_values")
        else:
            stage2_assignment = randomizer.assign_stage2_arm(case, R1)
            stage2_arm_id = stage2_assignment["arm"]

        stage2_arm = load_arm_config(stage2_arm_id, self._arms_dir)
        print(
            f"Stage 2 Arm: {stage2_arm_id} ({stage2_arm.get('name', '')}) "
            f"[Pool: {stage2_assignment['pool_used']}"
            f"{', forced' if stage2_assignment.get('forced') else ''}]\n"
        )

        logger.log_stage_transition(1, R1, stage2_arm_id, stage2_assignment)
        doctor.switch_arm(stage2_arm)
        print(f"--- STAGE 2 ({stage2_arm.get('name', '')}) ---")
        if adaptive_rounds:
            print(
                f"(rounds: adaptive, max={stage2_max}, min={stage2_min}, "
                f"coverage>={cov_cfg['threshold']:.0%}, check_every={cov_cfg['check_every']})"
            )

        arm_retrieval_enabled = (
            self._rag_enabled
            and bool((stage2_arm.get("tool_access") or {}).get("retrieval", False))
        )

        stage2_start = stage1_turns_used + 1
        last_stage2_turn = stage1_turns_used + stage2_max
        pending_retrieval_context: Optional[str] = None
        early_conclude = False
        stage2_early_stop: Optional[str] = None
        stage2_turn_idx = 0

        for turn in range(stage2_start, last_stage2_turn + 1):
            stage2_turn_idx += 1
            force_conclude = turn == last_stage2_turn or early_conclude
            allow = self._doctor_allow_finalize(
                coverage,
                cov_cfg,
                stage_turn_idx=stage2_turn_idx,
                stage_min_turns=stage2_min,
                force_conclude=force_conclude,
            )
            doctor_msg = doctor.respond(
                last_patient,
                force_conclude=force_conclude,
                retrieval_context=pending_retrieval_context,
                allow_finalize=allow,
            )
            if early_conclude and not doctor.has_concluded():
                stage2_early_stop = stage2_early_stop or "fact_coverage"
            early_conclude = False
            pending_retrieval_context = None

            if arm_retrieval_enabled:
                query = self._parse_retrieval_query(doctor_msg)
                if query:
                    if self._retriever is not None:
                        passages = self._retriever.retrieve(query, k=self._rag_k)
                        pending_retrieval_context = self._format_retrieval_result(passages)
                        print(f"[RAG] Query: {query!r} -> {len(passages)} passages retrieved")

            print(f"[Turn {turn}]")
            print(f"Doctor: {doctor_msg}")

            if doctor.has_concluded():
                if cov_cfg["enabled"]:
                    coverage.probe_pending(doctor.conversation_history, turn=turn)
                if stage2_early_stop is None and turn < last_stage2_turn:
                    stage2_early_stop = "doctor_concluded"
                logger.log_turn(
                    turn, 2, stage2_arm_id, doctor_msg, "",
                    mediq_meta=doctor.get_last_mediq_meta(),
                )
                print()
                break

            patient_facing_msg = self._strip_internal_markers(doctor_msg)
            last_patient = patient.respond(patient_facing_msg)
            print(f"Patient: {last_patient}\n")
            logger.log_turn(
                turn, 2, stage2_arm_id, doctor_msg, last_patient,
                mediq_meta=doctor.get_last_mediq_meta(),
            )

            if cov_cfg["enabled"]:
                probe = coverage.probe_if_due(turn, doctor.conversation_history)
                if probe:
                    print(
                        f"[Coverage] {probe['coverage_count']}/{probe['total_facts']} "
                        f"({probe['coverage_rate']:.0%})\n"
                    )

            if cov_cfg["enabled"]:
                coverage.probe_pending(doctor.conversation_history, turn=turn)

            if (
                adaptive_rounds
                and stage2_turn_idx >= stage2_min
                and coverage.meets_threshold()
                and turn < last_stage2_turn
            ):
                early_conclude = True
                stage2_early_stop = "fact_coverage"
                print(
                    f"[Adaptive] coverage {coverage.coverage_rate:.0%} — "
                    f"forcing diagnosis next turn\n"
                )

        logger._current["stage2_turns_used"] = stage2_turn_idx
        logger._current["stage2_turns_max"] = stage2_max
        if stage2_early_stop:
            logger._current["stage2_early_stop"] = stage2_early_stop

        cov_snap = self._finalize_coverage(logger, coverage, doctor, cov_cfg)
        print("--- Computing R2 (fact coverage) ---")
        R2 = self.judge.compute_R2_from_coverage(
            cov_snap["coverage_rate"],
            high_threshold=cov_cfg["r2_high_threshold"],
            coverage_count=cov_snap["coverage_count"],
            total_facts=cov_snap["total_facts"],
        )
        print(
            f"R2 Coverage: {R2['confidence_level']} (rate={R2['final_confidence']:.2f}, "
            f"source={R2.get('r2_source', '')})\n"
        )
        logger.log_R2(R2)

        print("--- Evaluating Outcome ---")
        final_diag = doctor.get_final_diagnosis()
        self._attach_mediq_encounter_fields(logger, doctor)
        outcome = self._evaluate_encounter_outcome(
            doctor=doctor, case=case, final_diag=final_diag, R2=R2,
        )

        trajectory = logger.finalize(outcome)
        self._print_outcome(
            trajectory,
            label=f"{trajectory['stage1_arm']} -> {trajectory['stage2_arm']}",
        )
        return trajectory

    def _print_outcome(self, trajectory: Dict[str, Any], label: str) -> None:
        outcome = trajectory.get("outcome") or {}
        print(f"Diagnosis Correct: {outcome.get('diag_correct')}")
        print(f"Red Flag Miss: {outcome.get('red_flag_miss')}")
        print(f"Dangerous Advice: {outcome.get('dangerous_advice')}")
        print(f"Turns Used: {trajectory.get('total_turns', 0)}")
        agg = self._aggregate_filename or f"{trajectory['case_id']}.jsonl"
        print(f"\n{'=' * 60}")
        print(f"Trajectory: {label}")
        print(f"Saved to: {self.output_path}")
        print(f"{'=' * 60}\n")

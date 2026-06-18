"""Closed-loop adaptive eval: refit Q every N encounters, biased-random arm assignment."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from smart_trial.core.orchestrator import TrialOrchestrator, resolve_path
from smart_trial.eval.adaptive_policy import AdaptivePolicyAssigner
from smart_trial.eval.case_lists import CLOSED_LOOP_CASE_IDS
from smart_trial.eval.resume import load_completed_adaptive_case_ids
from smart_trial.q_learning import load as ql_load
from smart_trial.q_learning import q_learning
from smart_trial.trajectory_log.trajectory_logger import TrajectoryLogger


def _aggregate_path(orch: TrialOrchestrator) -> Path:
    name = orch.config.get("logging", {}).get("aggregate_filename")
    if not name:
        raise SystemExit("logging.aggregate_filename is required for eval batch runs.")
    return orch._output_dir / name


def _load_encounters(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return TrajectoryLogger.load_aggregate(str(path))


def fit_policy(
    grid_encounters: List[Dict[str, Any]],
    adaptive_encounters: List[Dict[str, Any]],
) -> q_learning.QLearningResult:
    combined = list(grid_encounters) + list(adaptive_encounters)
    df = ql_load.to_dataframe(combined)
    if df.empty:
        raise ValueError("no valid encounters to fit Q-learning policy")
    return q_learning.fit(df)


def restore_loop_state(
    run_cfg: dict,
    orch: TrialOrchestrator,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], AdaptivePolicyAssigner, int]:
    """Load grid seed data, completed adaptive rows, and current policy assigner."""
    initial_path = resolve_path(str(run_cfg.get("initial_q_from", "")))
    grid_rows = _load_encounters(initial_path)
    if not grid_rows:
        raise SystemExit(f"initial_q_from missing or empty: {initial_path}")

    agg = _aggregate_path(orch)
    adaptive_rows = [
        r for r in _load_encounters(agg)
        if r.get("run_mode") == "smart_adaptive_loop"
    ]

    refit_every = max(1, int(run_cfg.get("refit_every_n", 5)))
    refit_generation = len(adaptive_rows) // refit_every
    result = fit_policy(grid_rows, adaptive_rows)
    temperature = float(run_cfg.get("policy_temperature", 1.0))
    assigner = AdaptivePolicyAssigner(
        result,
        refit_generation=refit_generation,
        temperature=temperature,
    )
    return grid_rows, adaptive_rows, assigner, refit_generation


def run_adaptive_loop(
    orch: TrialOrchestrator,
    cases: List[dict],
    seed: int,
    resume: bool,
) -> int:
    run_cfg = orch.config.get("run") or {}
    refit_every = max(1, int(run_cfg.get("refit_every_n", 5)))
    burn_in = max(0, int(run_cfg.get("burn_in", 0)))
    temperature = float(run_cfg.get("policy_temperature", 1.0))

    grid_rows, adaptive_rows, assigner, refit_generation = restore_loop_state(run_cfg, orch)
    done = load_completed_adaptive_case_ids(_aggregate_path(orch)) if resume else set()
    since_refit = len(adaptive_rows) % refit_every

    ran = 0
    for case in cases:
        cid = case["case_id"]
        if cid in done:
            print(f"[skip] adaptive already done: {cid}")
            continue

        n_completed_before = len(adaptive_rows)
        use_random = burn_in > 0 and n_completed_before < burn_in
        if use_random:
            print(
                f"[burn-in] {cid} ({n_completed_before + 1}/{burn_in}) — uniform SMART randomization"
            )
            traj = orch.run_encounter(case, seed=seed, run_mode="smart_random")
        else:
            traj = orch.run_encounter(
                case,
                seed=seed,
                run_mode="smart_adaptive_loop",
                policy_assigner=assigner,
            )

        adaptive_rows.append(traj)
        done.add(cid)
        ran += 1
        since_refit += 1

        if since_refit >= refit_every:
            refit_generation += 1
            result = fit_policy(grid_rows, adaptive_rows)
            assigner = AdaptivePolicyAssigner(
                result,
                refit_generation=refit_generation,
                temperature=temperature,
            )
            since_refit = 0
            print(
                f"[refit] generation={refit_generation} "
                f"after {len(adaptive_rows)} adaptive encounter(s)"
            )

    return ran


def resolve_closed_loop_case_ids(cfg: dict) -> List[str]:
    eval_cfg = cfg.get("eval") or {}
    explicit = eval_cfg.get("case_ids")
    if explicit:
        return list(explicit)
    return list(CLOSED_LOOP_CASE_IDS)

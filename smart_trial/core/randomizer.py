import hashlib
import random
from typing import Any, Dict, List


def _stable_seed(global_seed: int, case_id: str, tag: str) -> int:
    """Deterministic per-case RNG seed (avoids Python's salted hash())."""
    digest = hashlib.sha256(f"{global_seed}:{case_id}:{tag}".encode()).hexdigest()
    return int(digest[:16], 16)


class TrialRandomizer:
    """
    SMART re-randomization: stage-2 pool depends on R1 responder status;
    stage-3 pool depends on R2 confidence level only (not correctness).
    """

    STAGE2_POOLS = {
        "responder": ["A2a", "A2b", "A2c"],
        "non-responder": ["A2a", "A2b"],
    }

    STAGE3_POOLS = {
        "high": ["A3a", "A3b"],
        "low": ["A3b", "A3c"],
    }

    STAGE1_POOL = ["A1a", "A1b", "A1c"]

    def __init__(self, seed: int, stratify_by: str = "case_category"):
        self.seed = seed
        self.stratify_by = stratify_by

    def assign_stage1_arm(self, case: Dict[str, Any]) -> str:
        case_seed = _stable_seed(self.seed, case["case_id"], "stage1")
        rng = random.Random(case_seed)
        return rng.choice(self.STAGE1_POOL)

    def assign_stage2_arm(self, case: Dict[str, Any], R1: Dict[str, Any]) -> Dict[str, Any]:
        is_responder = bool(R1.get("responder", False))
        pool_key = "responder" if is_responder else "non-responder"
        pool = self.STAGE2_POOLS[pool_key]
        case_seed = _stable_seed(self.seed, case["case_id"], "stage2")
        rng = random.Random(case_seed)
        arm = rng.choice(pool)
        return {
            "arm": arm,
            "pool_used": pool_key,
            "pool": pool,
            "R1_total": R1.get("total", 0),
        }

    def assign_stage3_arm(self, case: Dict[str, Any], R2: Dict[str, Any]) -> Dict[str, Any]:
        confidence_level = R2.get("confidence_level", "low")
        pool = self.STAGE3_POOLS.get(confidence_level, ["A3b", "A3c"])
        case_seed = _stable_seed(self.seed, case["case_id"], "stage3")
        rng = random.Random(case_seed)
        arm = rng.choice(pool)
        return {
            "arm": arm,
            "pool_used": confidence_level,
            "pool": pool,
            "R2_confidence": R2.get("final_confidence"),
        }

    def get_assignment_summary(
        self,
        stage1_result: str,
        stage2_result: Dict[str, Any],
        stage3_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "stage1_arm": stage1_result,
            "stage2_arm": stage2_result["arm"],
            "stage2_pool": stage2_result["pool_used"],
            "stage3_arm": stage3_result["arm"],
            "stage3_pool": stage3_result["pool_used"],
            "trajectory_id": f"{stage1_result}->{stage2_result['arm']}->{stage3_result['arm']}",
        }

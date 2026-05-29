"""Constants and paths shared across the Q-learning analysis package."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DEFAULT_ENCOUNTERS_DIR = PROJECT_ROOT / "smart_trial" / "outputs" / "encounters"
DEFAULT_RESULTS_DIR = PACKAGE_ROOT / "outputs"

STAGE1_ARMS = ["A1a", "A1b", "A1c"]

STAGE2_POOLS = {
    "responder": ["A2a", "A2b", "A2c"],
    "non-responder": ["A2a", "A2b"],
}

STAGE3_POOLS = {
    "high": ["A3a", "A3b"],
    "low": ["A3b", "A3c"],
}

R1_RESPONDER_THRESHOLD = 6
R2_HIGH_CONFIDENCE_THRESHOLD = 0.7

OUTCOME_KEY = "diag_correct"

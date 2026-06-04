"""Constants and paths shared across the Q-learning analysis package."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

DEFAULT_ENCOUNTERS_DIR = PROJECT_ROOT / "smart_trial" / "outputs" / "encounters"
DEFAULT_ENCOUNTERS_FILE = PROJECT_ROOT / "smart_trial" / "outputs" / "encounters.jsonl"
DEFAULT_RESULTS_DIR = PACKAGE_ROOT / "outputs"

STAGE1_ARMS = ["A1a", "A1b", "A1c"]
STAGE2_ARMS = ["A2a", "A2b", "A2c"]
STAGE3_ARMS = ["A3a", "A3b", "A3c"]

STAGE2_POOLS = {
    "responder": ["A2a", "A2b", "A2c"],
    "non-responder": ["A2a", "A2b"],
}

STAGE3_POOLS = {
    "high": ["A3a", "A3b"],
    "low": ["A3b", "A3c"],
}

# Reference arms for identifiable regression (contrast coding).
REFERENCE_ARMS = {
    "stage1": "A1a",
    "stage2": "A2a",
    "stage3": "A3b",
}

R1_RESPONDER_THRESHOLD = 6
R2_HIGH_CONFIDENCE_THRESHOLD = 0.7

OUTCOME_KEY = "diag_correct"

# Map raw case_category values from encounter logs → feature levels.
CATEGORY_MAP = {
    "Cardiology": "Cardiology",
    "Pulmonology": "Pulmonology",
    "Gastroenterology": "Gastroenterology",
    "Neurology": "Neurology",
    "Neuro": "Neurology",
    "Infectious_Disease": "Infectious_Disease",
    "Infectious Disease": "Infectious_Disease",
    "Endocrinology": "Endocrinology",
    "Pediatrics": "Other",
    "Psychiatry": "Other",
    "Other": "Other",
}

CATEGORY_LEVELS = [
    "Cardiology",
    "Pulmonology",
    "Gastroenterology",
    "Neurology",
    "Infectious_Disease",
    "Endocrinology",
    "Other",
]

# Default Ridge strength; override via fit(..., ridge_alpha=...).
DEFAULT_RIDGE_ALPHA = 1.0

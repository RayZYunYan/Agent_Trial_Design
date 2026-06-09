from smart_trial.q_learning.load import _resolve_literacy_id


def test_literacy_from_main_persona_register():
    enc = {
        "persona": {
            "persona_id": "random",
            "vocabulary_register": "interactive",
            "jargon_comprehension": "interactive",
        }
    }
    assert _resolve_literacy_id(enc) == "literacy_I"


def test_literacy_from_fixed_persona_id():
    enc = {"persona": {"persona_id": "literacy_F", "vocabulary_register": "functional"}}
    assert _resolve_literacy_id(enc) == "literacy_F"


def test_literacy_from_preset_alias():
    enc = {"persona": {"persona_id": "low_literacy", "vocabulary_register": "functional"}}
    assert _resolve_literacy_id(enc) == "literacy_F"


def test_literacy_from_legacy_literacy_persona_block():
    enc = {
        "literacy_persona": {
            "persona_id": "literacy_I",
            "axes": {"vocabulary_register": "I"},
        }
    }
    assert _resolve_literacy_id(enc) == "literacy_I"


def test_literacy_from_legacy_language_level_basic():
    enc = {
        "persona": {
            "personality": "distrustful",
            "language_level": "basic",
            "recall": "high",
            "emotional_state": "evasive",
        }
    }
    assert _resolve_literacy_id(enc) == "literacy_F"


def test_literacy_from_legacy_language_level_intermediate():
    enc = {"persona": {"language_level": "intermediate"}}
    assert _resolve_literacy_id(enc) == "literacy_I"


def test_literacy_from_legacy_language_level_advanced():
    enc = {"persona": {"language_level": "advanced"}}
    assert _resolve_literacy_id(enc) == "literacy_C"


def test_literacy_from_jargon_comprehension_fallback():
    enc = {"persona": {"persona_id": "random", "jargon_comprehension": "critical"}}
    assert _resolve_literacy_id(enc) == "literacy_C"


def test_load_encounters_jsonl_literacy_not_null():
    from pathlib import Path

    path = Path("smart_trial/outputs/encounters.jsonl")
    if not path.is_file():
        return
    from smart_trial.q_learning.load import load

    df = load(path)
    assert len(df) > 0
    assert df["literacy_id"].notna().all()

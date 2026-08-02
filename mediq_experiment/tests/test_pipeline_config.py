"""Offline checks for mediq_experiment config / providers / MediQ CLI flags."""
from __future__ import annotations

from pathlib import Path

import pytest

from mediq_experiment.io_utils import ROOT, load_yaml
from mediq_experiment.model_chat import api_use_flag, is_local_provider, resolve_provider
from mediq_experiment.run_mediq import build_mediq_cmd
from mediq_experiment.run_pipeline import _doctor_letter, _list_doctors


CONFIG = ROOT / "mediq_experiment" / "config.yaml"
SMOKE = ROOT / "mediq_experiment" / "config_smoke.yaml"


@pytest.fixture(scope="module")
def full_cfg():
    assert CONFIG.exists()
    return load_yaml(CONFIG)


@pytest.fixture(scope="module")
def smoke_cfg():
    assert SMOKE.exists()
    return load_yaml(SMOKE)


def test_full_config_has_five_doctors(full_cfg):
    docs = _list_doctors(full_cfg)
    assert [k for k, _ in docs] == [
        "doctor_a",
        "doctor_b",
        "doctor_c",
        "doctor_d",
        "doctor_e",
    ]


def test_doctor_c_is_luna_api(full_cfg):
    block = full_cfg["models"]["doctor_c"]
    assert "luna" in block["name"].lower()
    assert resolve_provider(block) == "openai"
    assert api_use_flag("openai") == "openai"
    assert not is_local_provider("openai")


def test_local_doctors_de(full_cfg):
    for key in ("doctor_d", "doctor_e"):
        p = resolve_provider(full_cfg["models"][key])
        assert is_local_provider(p)
        assert api_use_flag(p) is None
    assert "Qwen3.5-4B" in full_cfg["models"]["doctor_d"]["name"]
    assert "Llama-3.1-8B" in full_cfg["models"]["doctor_e"]["name"]
    assert "mlx-community" in full_cfg["models"]["doctor_d"]["mlx_name"]
    assert "mlx-community" in full_cfg["models"]["doctor_e"]["mlx_name"]
    assert full_cfg["mediq"].get("use_mlx") is True


def test_api_doctors_abc(full_cfg):
    assert resolve_provider(full_cfg["models"]["doctor_a"]) == "openai"
    assert resolve_provider(full_cfg["models"]["doctor_b"]) == "anthropic"
    assert resolve_provider(full_cfg["models"]["doctor_c"]) == "openai"


def test_smoke_isolated_output_and_one_case(smoke_cfg, full_cfg):
    assert smoke_cfg["data"]["max_cases"] == 1
    assert smoke_cfg["pipeline"]["output_dir"] == "mediq_experiment/outputs_smoke"
    assert smoke_cfg["models"]["doctor_c"]["name"] == full_cfg["models"]["doctor_c"]["name"]
    assert smoke_cfg["mediq"].get("use_vllm") is False


def test_build_cmd_api_vs_local(full_cfg):
    mediq = full_cfg["mediq"]
    api_cmd = build_mediq_cmd(
        expert_model="gpt-5.6-luna",
        patient_model="claude-haiku-4-5",
        data_dir=Path("."),
        dev_filename="x.jsonl",
        output_filename=Path("o.jsonl"),
        log_dir=Path("logs"),
        mediq_cfg=mediq,
        expert_provider="openai",
    )
    assert "--use_api" in api_cmd
    assert "--use_vllm" not in api_cmd

    local_cmd = build_mediq_cmd(
        expert_model="Qwen/Qwen3.5-4B",
        patient_model="claude-haiku-4-5",
        data_dir=Path("."),
        dev_filename="x.jsonl",
        output_filename=Path("o.jsonl"),
        log_dir=Path("logs"),
        mediq_cfg=mediq,
        expert_provider="huggingface",
    )
    assert "--use_api" not in local_cmd
    assert "--use_vllm" not in local_cmd
    assert "--use_mlx" in local_cmd


def test_doctor_letter():
    assert _doctor_letter("doctor_a") == "a"
    assert _doctor_letter("doctor_e") == "e"


def test_local_hf_ids(full_cfg):
    for key in ("doctor_d", "doctor_e"):
        assert "/" in full_cfg["models"][key]["name"]

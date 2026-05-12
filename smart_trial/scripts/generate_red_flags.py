"""
Offline LLM pass: generate red flags per case and save JSON cache for loader.apply_red_flag_cache.
Usage (from repo root):
  python -m smart_trial.scripts.generate_red_flags --out smart_trial/data/red_flag_cache.json --max-cases 50
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from smart_trial.data.loader import load_cases_from_config
from smart_trial.models.model_client import ModelClient

PROMPT = """给定病人主诉，列出这个主诉最重要的3-5个危险信号（red flags）。
这些是医生必须问到的、可能提示严重疾病的症状或体征。

主诉：{chief_complaint}
病人年龄：{age}，性别：{gender}

请只输出JSON格式，不要有其他文字：
{{"red_flags": ["red flag 1", "red flag 2", ...]}}"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="smart_trial/data/red_flag_cache.json")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config else PROJECT_ROOT / "smart_trial" / "config" / "trial_config.yaml"
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cases = load_cases_from_config(cfg)
    if args.max_cases:
        cases = cases[: args.max_cases]

    mc_cfg = cfg["models"]["patient_simulator"]
    client = ModelClient(mc_cfg["provider"], mc_cfg["model_name"], temperature=float(mc_cfg.get("temperature", 0.3)))

    cache: dict = {}
    for case in cases:
        prompt = PROMPT.format(
            chief_complaint=case.get("chief_complaint", ""),
            age=case.get("age", "unknown"),
            gender=case.get("gender", "unknown"),
        )
        response = client.chat([{"role": "user", "content": prompt}], temperature=0.2)
        flags: list = []
        try:
            clean = re.sub(r"```json\s*|\s*```", "", response).strip()
            data = json.loads(clean)
            flags = data.get("red_flags", []) or []
        except json.JSONDecodeError:
            flags = []
        cache[case["case_id"]] = flags
        print(f"Processed {case['case_id']}")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(cache)} entries to {out_path}")


if __name__ == "__main__":
    main()

"""End-of-case fact coverage on MediQ dialogue JSONL (no mid-dialogue probes)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from mediq_experiment.io_utils import (
    ROOT,
    accuracy_summary,
    load_yaml,
    normalize_facts,
    read_jsonl,
    transcript_to_history,
    write_jsonl,
)
from smart_trial.core.judge import StageJudge, filter_scorable_atomic_facts
from smart_trial.models.model_client import ModelClient


def _case_from_mediq_row(row: Dict[str, Any]) -> Dict[str, Any]:
    info = row.get("info") or {}
    facts = normalize_facts(info.get("facts"))
    initial = (info.get("initial_info") or "").strip()
    return {
        "case_id": str(row.get("id")),
        "chief_complaint": initial or str(info.get("question") or ""),
        "atomic_facts": facts,
        "ground_truth_answer": info.get("correct_answer"),
        "ground_truth_idx": info.get("correct_answer_idx"),
        "options": info.get("options") or {},
    }


def score_row(judge: StageJudge, row: Dict[str, Any]) -> Dict[str, Any]:
    interactive = row.get("interactive_system") or {}
    case = _case_from_mediq_row(row)
    facts = filter_scorable_atomic_facts(case.get("atomic_facts") or [])
    history = transcript_to_history(
        list(interactive.get("questions") or []),
        list(interactive.get("answers") or []),
    )
    letter = interactive.get("letter_choice")
    gold = case.get("ground_truth_idx")
    final_acc = bool(letter is not None and gold is not None and str(letter) == str(gold))

    if not facts:
        coverage = {
            "coverage_rate": None,
            "coverage_count": 0,
            "total_facts": 0,
            "covered_facts": [],
            "note": "no facts available on this case",
        }
    elif not history:
        coverage = {
            "coverage_rate": 0.0,
            "coverage_count": 0,
            "total_facts": len(facts),
            "covered_facts": [],
            "note": "empty transcript",
        }
    else:
        delta = judge.probe_fact_coverage_incremental(
            case,
            new_conversation=history,
            atomic_facts=facts,
            already_covered=[],
            review_mode=True,
        )
        newly = delta.get("newly_covered") or []
        coverage = {
            "coverage_rate": round(len(newly) / len(facts), 4) if facts else 0.0,
            "coverage_count": len(newly),
            "total_facts": len(facts),
            "covered_facts": newly,
        }

    out = dict(row)
    out["eval"] = {
        "final_accuracy": final_acc,
        "letter_choice": letter,
        "correct_answer_idx": gold,
        "fact_coverage": coverage,
    }
    return out


def score_file(
    input_path: Path,
    output_path: Path,
    *,
    judge_provider: str,
    judge_model: str,
    judge_temperature: float = 0.1,
    skip_if_exists: bool = False,
) -> Dict[str, Any]:
    load_dotenv(ROOT / ".env")
    if skip_if_exists and output_path.exists() and output_path.stat().st_size > 0:
        scored = read_jsonl(output_path)
        rates = [
            float(r["eval"]["fact_coverage"]["coverage_rate"])
            for r in scored
            if r.get("eval", {}).get("fact_coverage", {}).get("coverage_rate") is not None
        ]
        summary = {
            "n": len(scored),
            "final_accuracy": accuracy_summary(
                [{"correct": r["eval"]["final_accuracy"]} for r in scored]
            ),
            "mean_fact_coverage": (sum(rates) / len(rates)) if rates else None,
            "input": str(input_path),
            "output": str(output_path),
            "skipped": True,
        }
        print(f"  skip coverage (exists): {output_path}")
        return summary

    judge = StageJudge(ModelClient(judge_provider, judge_model, temperature=judge_temperature))
    rows = read_jsonl(input_path)
    scored = [score_row(judge, row) for row in rows]
    write_jsonl(output_path, scored)

    rates = [
        float(r["eval"]["fact_coverage"]["coverage_rate"])
        for r in scored
        if r.get("eval", {}).get("fact_coverage", {}).get("coverage_rate") is not None
    ]
    summary = {
        "n": len(scored),
        "final_accuracy": accuracy_summary(
            [{"correct": r["eval"]["final_accuracy"]} for r in scored]
        ),
        "mean_fact_coverage": (sum(rates) / len(rates)) if rates else None,
        "input": str(input_path),
        "output": str(output_path),
        "skipped": False,
    }
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Score MediQ JSONL with end-of-case fact coverage.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "mediq_experiment" / "config.yaml")
    parser.add_argument("--judge-provider", default=None)
    parser.add_argument("--judge-model", default=None)
    args = parser.parse_args(argv)

    cfg = load_yaml(args.config) if args.config.exists() else {}
    judge_cfg = (cfg.get("models") or {}).get("judge") or {}
    provider = args.judge_provider or judge_cfg.get("provider", "openai")
    model = args.judge_model or judge_cfg.get("name", "gpt-4o-mini")
    temp = float(judge_cfg.get("temperature", 0.1))

    summary = score_file(
        args.input,
        args.output,
        judge_provider=provider,
        judge_model=model,
        judge_temperature=temp,
    )
    print(summary)


if __name__ == "__main__":
    main()

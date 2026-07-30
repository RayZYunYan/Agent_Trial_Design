"""Cross-model finalize: frozen MediQ transcript → other model answers MCQ once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from dotenv import load_dotenv

from mediq_experiment.io_utils import (
    ROOT,
    accuracy_summary,
    extract_letter,
    format_options,
    format_transcript,
    load_yaml,
    read_jsonl,
    write_jsonl,
)
from mediq_experiment.model_chat import chat as model_chat

CROSS_PROMPT = """You are a medical expert. Another doctor already interviewed the patient.
Based ONLY on the dialogue below, choose the single best answer to the multiple-choice question.
Reply with ONLY the letter (e.g. A, B, C, or D). Do not ask questions.

Question: {question}

Options:
{options}

Dialogue:
{transcript}
"""


def finalize_from_transcript(
    chat_fn: Callable[..., str],
    row: Dict[str, Any],
    *,
    responder_model: str,
) -> Dict[str, Any]:
    interactive = row.get("interactive_system") or {}
    info = row.get("info") or {}
    options = info.get("options") or {}
    transcript = format_transcript(
        list(interactive.get("questions") or []),
        list(interactive.get("answers") or []),
        initial_info=str(info.get("initial_info") or ""),
    )
    prompt = CROSS_PROMPT.format(
        question=info.get("question") or "",
        options=format_options(options),
        transcript=transcript or "(empty dialogue)",
    )
    raw = chat_fn([{"role": "user", "content": prompt}])
    letter = extract_letter(raw, valid=options.keys())
    gold = info.get("correct_answer_idx")
    correct = bool(letter is not None and gold is not None and str(letter) == str(gold))
    return {
        "id": row.get("id"),
        "source_doctor_letter": interactive.get("letter_choice"),
        "source_doctor_correct": interactive.get("correct"),
        "responder_model": responder_model,
        "responder_letter": letter,
        "responder_raw": raw,
        "correct_answer_idx": gold,
        "correct": correct,
        "num_questions": interactive.get("num_questions"),
    }


def run_cross(
    source_jsonl: Path,
    output_jsonl: Path,
    *,
    responder_provider: str,
    responder_model: str,
    use_vllm: bool = False,
    load_in_4bit: bool = False,
    max_tokens: int = 64,
) -> Dict[str, Any]:
    load_dotenv(ROOT / ".env")

    def _chat(messages: List[Dict[str, str]]) -> str:
        return model_chat(
            provider=responder_provider,
            model_name=responder_model,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            use_vllm=use_vllm,
            load_in_4bit=load_in_4bit,
        )

    rows = read_jsonl(source_jsonl)
    results = [
        finalize_from_transcript(_chat, row, responder_model=responder_model) for row in rows
    ]
    write_jsonl(output_jsonl, results)
    summary = accuracy_summary(results)
    summary.update(
        {
            "source": str(source_jsonl),
            "output": str(output_jsonl),
            "responder_model": responder_model,
            "responder_provider": responder_provider,
        }
    )
    return summary


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Cross-model MCQ finalize from MediQ transcripts.")
    parser.add_argument("--source", required=True, type=Path, help="MediQ results JSONL (dialogue donor)")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--responder-provider", default="openai")
    parser.add_argument("--responder-model", required=True)
    parser.add_argument("--use-vllm", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--config", type=Path, default=ROOT / "mediq_experiment" / "config.yaml")
    args = parser.parse_args(argv)

    if args.config.exists():
        load_yaml(args.config)  # validate readable

    summary = run_cross(
        args.source,
        args.output,
        responder_provider=args.responder_provider,
        responder_model=args.responder_model,
        use_vllm=args.use_vllm,
        load_in_4bit=args.load_in_4bit,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

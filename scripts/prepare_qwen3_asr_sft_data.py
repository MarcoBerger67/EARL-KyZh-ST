from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Convert compact ky->zh JSONL into Qwen3-ASR official SFT JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root_dir / "data" / "train_ky2zh_with_ky_text.jsonl",
        help="Compact JSONL with id/audio/prompt/source_text/text_zh fields.",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=root_dir / "data" / "qwen3_asr_sft_train.jsonl",
    )
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=root_dir / "data" / "qwen3_asr_sft_eval.jsonl",
    )
    parser.add_argument("--audio-prefix-from", type=str, default="")
    parser.add_argument("--audio-prefix-to", type=str, default="")
    parser.add_argument("--target-field", choices=["source_text", "text_zh"], default="source_text")
    parser.add_argument(
        "--language",
        type=str,
        default="None",
        help='Qwen3-ASR language prefix. Use "None" for Kyrgyz because it is not in the official supported-language list.',
    )
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def rewrite_path(path: str, prefix_from: str, prefix_to: str) -> str:
    if not prefix_from:
        return path
    if path.startswith(prefix_to):
        return path
    if not path.startswith(prefix_from):
        raise ValueError(f"Path {path!r} does not start with {prefix_from!r}.")
    return prefix_to + path.removeprefix(prefix_from)


def load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            audio = str(record.get("audio", "") or "")
            target = normalize_text(record.get(args.target_field, ""))
            if not audio or not target:
                continue
            rows.append(
                {
                    "id": str(record.get("id", f"line_{line_number}")),
                    "audio": rewrite_path(audio, args.audio_prefix_from, args.audio_prefix_to),
                    "text": f"language {args.language}<asr_text>{target}",
                }
            )
            if args.limit is not None and len(rows) >= args.limit:
                break
    if not rows:
        raise ValueError(f"No usable rows loaded from {args.input}.")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    if not (0.0 < args.val_ratio < 0.5):
        raise ValueError("--val-ratio must be between 0 and 0.5.")
    rows = load_rows(args)
    random.Random(args.seed).shuffle(rows)
    val_count = max(1, int(round(len(rows) * args.val_ratio)))
    eval_rows = rows[:val_count]
    train_rows = rows[val_count:]
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.eval_output, eval_rows)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "train_output": str(args.train_output),
                "eval_output": str(args.eval_output),
                "target_field": args.target_field,
                "language": args.language,
                "train_rows": len(train_rows),
                "eval_rows": len(eval_rows),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

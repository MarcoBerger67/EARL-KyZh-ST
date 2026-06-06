from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_PROMPT = (
    "Translate the following speech segment into chinese. Follow these specific "
    "instructions for formatting the answer:\n"
    "* Only output the translation, with no newlines.\n"
    "* When translating numbers, write the digits, i.e. write 1.7 and not one "
    "point seven, and write 3 instead of three."
)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Convert compact ky->zh JSONL into Gemma4 audio SFT messages JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=root_dir / "data" / "train_ky2zh_with_ky_text.jsonl",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=root_dir / "data" / "gemma4_sft_train_from_compact.jsonl",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=root_dir / "data" / "gemma4_sft_val_from_compact.jsonl",
    )
    parser.add_argument("--audio-prefix-from", type=str, default="")
    parser.add_argument("--audio-prefix-to", type=str, default="")
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


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            audio = str(record.get("audio", "") or "")
            target = normalize_text(record.get("text_zh", ""))
            if not audio or not target:
                continue
            prompt = str(record.get("prompt", "") or "").strip() or DEFAULT_PROMPT
            rows.append(
                {
                    "id": str(record.get("id", f"line_{line_number}")),
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "audio",
                                    "path": rewrite_path(
                                        audio,
                                        args.audio_prefix_from,
                                        args.audio_prefix_to,
                                    ),
                                },
                                {"type": "text", "text": prompt},
                            ],
                        },
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": target}],
                        },
                    ],
                    "source_text": normalize_text(record.get("source_text", "")),
                }
            )
            if args.limit is not None and len(rows) >= args.limit:
                break
    if not rows:
        raise ValueError(f"No usable rows loaded from {args.input}.")
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.val_output, val_rows)
    print(
        json.dumps(
            {
                "input": str(args.input),
                "train_output": str(args.train_output),
                "val_output": str(args.val_output),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "val_ratio": args.val_ratio,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()

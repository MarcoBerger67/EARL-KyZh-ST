from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from eval_suite.runner import run_eval_spec


VALID_LABELS = {"PER", "LOC", "ORG", "TIME", "NUM", "TERM", "TITLE"}

ENTITY_FOCUSED_TRANSLATION_PROMPT = (
    "You are a strict Kyrgyz-to-Chinese speech translation system.\n\n"
    "Task: translate the input Kyrgyz speech segment into fluent Simplified Chinese.\n\n"
    "Output rules:\n"
    "1. Output only the final Simplified Chinese translation.\n"
    "2. Output one single line only.\n"
    "3. Do not output Kyrgyz, English, transliteration, explanations, notes, labels, speaker tags, or markdown.\n"
    "4. Do not output prefixes or suffixes such as \"Translation:\", \"Answer:\", or quotation marks around the answer.\n"
    "5. Translate the complete speech segment; do not stop after only the beginning and do not omit content.\n\n"
    "Entity fidelity rules. Pay special attention to preserving and translating entities accurately:\n"
    "- PER: person names, speakers, authors, historical figures, and transliterated names. Preserve the full name when audible; do not drop given names, family names, or middle dots such as \"·\".\n"
    "- LOC: countries, cities, regions, landmarks, buildings, roads, facilities, and natural geographic names. Translate or transliterate them into the most natural Chinese form.\n"
    "- ORG: organizations, institutions, companies, schools, government departments, media outlets, hospitals, teams, and international bodies. Preserve the organization as a named unit.\n"
    "- TERM: domain terms, technical concepts, event names, methods, diseases, products, policies, and important specialized expressions. Use precise Chinese terminology when possible.\n"
    "- NUM: numbers, quantities, percentages, money, rankings, temperatures, measurements, and identifiers. Write digits using Arabic numerals, e.g. 1.7, 3, 80%.\n"
    "- TIME: dates, years, periods, durations, festivals, and time expressions. Preserve the time meaning accurately.\n\n"
    "Faithfulness rules:\n"
    "1. Preserve names, places, organizations, terms, numbers, and time expressions as much as possible.\n"
    "2. Do not summarize, rewrite, expand, or invent content not supported by the speech.\n"
    "3. If some audio is unclear, translate conservatively based on the audio and context; do not hallucinate.\n"
)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate a Gemma-4 base audio model on testt.jsonl with BLEU/chrF/entity recall."
    )
    parser.add_argument(
        "--test-data-path",
        type=Path,
        default=root_dir / "data" / "converted_testt_format" / "testt.jsonl",
    )
    parser.add_argument(
        "--test-entity-path",
        type=Path,
        default=root_dir / "data" / "converted_testt_format" / "testt.ner.jsonl",
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=str(root_dir / "model" / "gemma-4-e2b-it"),
    )
    parser.add_argument(
        "--processor-path",
        type=str,
        default=None,
        help="Optional processor path. Use this when the base model directory lacks processor/tokenizer files.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional LoRA adapter path, e.g. SFT+GRPO adapter_best.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Name stored in metrics. Defaults to the base model directory name.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root_dir / "eval_outputs" / "gemma4_base_testt",
    )
    parser.add_argument("--audio-prefix-from", type=str, default="")
    parser.add_argument("--audio-prefix-to", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device-map", type=str, default="none")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--prompt-mode",
        choices=["entity_focused", "dataset", "fixed"],
        default="entity_focused",
        help="entity_focused uses a strict translation prompt with explicit entity fidelity rules.",
    )
    parser.add_argument(
        "--prompt-text",
        type=str,
        default=ENTITY_FOCUSED_TRANSLATION_PROMPT,
        help="Prompt used when --prompt-mode is entity_focused or fixed.",
    )
    parser.add_argument(
        "--entity-lcs-labels",
        type=str,
        default="LOC,PER,TERM,NUM,ORG,TIME",
        help="Comma-separated labels reported separately for entity recall.",
    )
    parser.add_argument(
        "--entity-lcs-skip-labels",
        type=str,
        default="",
        help="Comma-separated labels skipped in overall entity recall. Use an empty string to include all labels.",
    )
    parser.add_argument(
        "--entity-metric",
        choices=["lcs", "hard_key_recall"],
        default="hard_key_recall",
        help="Entity metric. hard_key_recall requires the full normalized reference entity to appear in prediction.",
    )
    parser.add_argument("--config-out", type=Path, default=None)
    return parser.parse_args()


def record_id(record: dict[str, Any], fallback: str) -> str:
    return str(record.get("id") or record.get("key") or fallback)


def validate_testt_sidecar(data_path: Path, entity_path: Path, limit: int | None) -> dict[str, Any]:
    if not data_path.exists():
        raise FileNotFoundError(f"Missing test data: {data_path}")
    if not entity_path.exists():
        raise FileNotFoundError(f"Missing entity sidecar: {entity_path}")

    data_ids: list[str] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            data_ids.append(record_id(json.loads(line), f"line_{line_number}"))
            if limit is not None and len(data_ids) >= limit:
                break

    entity_rows = 0
    entity_ids: list[str] = []
    empty_entity_rows = 0
    malformed_rows = 0
    invalid_entities = 0
    entity_count = 0
    label_counts: dict[str, int] = {}
    with entity_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            if limit is not None and entity_rows >= limit:
                break
            entity_rows += 1
            record = json.loads(line)
            entity_ids.append(record_id(record, f"line_{line_number}"))
            entities = record.get("entities")
            if not isinstance(entities, list):
                malformed_rows += 1
                entities = []
            if not entities:
                empty_entity_rows += 1
            for item in entities:
                if not isinstance(item, dict):
                    invalid_entities += 1
                    continue
                text = str(item.get("text", "") or "").strip()
                label = str(item.get("label", "") or "").strip().upper()
                if not text or label not in VALID_LABELS:
                    invalid_entities += 1
                    continue
                entity_count += 1
                label_counts[label] = label_counts.get(label, 0) + 1

    if len(data_ids) != entity_rows:
        raise ValueError(
            f"Line count mismatch under limit={limit}: test rows={len(data_ids)}, sidecar rows={entity_rows}"
        )
    mismatches = [
        {"index": index + 1, "test_id": left, "entity_id": right}
        for index, (left, right) in enumerate(zip(data_ids, entity_ids))
        if left != right
    ]
    if mismatches:
        raise ValueError(f"testt.jsonl and sidecar ids are not aligned. First mismatches: {mismatches[:5]}")
    if malformed_rows or invalid_entities:
        raise ValueError(
            f"Malformed entity sidecar: malformed_rows={malformed_rows}, invalid_entities={invalid_entities}"
        )

    return {
        "rows": len(data_ids),
        "entity_rows": entity_rows,
        "empty_entity_rows": empty_entity_rows,
        "entity_count": entity_count,
        "label_counts": dict(sorted(label_counts.items())),
        "empty_entity_rows_are_kept": True,
    }


def build_eval_config(args: argparse.Namespace) -> dict[str, Any]:
    model_name = args.model_name or Path(args.base_model_path.rstrip("/")).name
    entity_lcs_labels = [label.strip().upper() for label in args.entity_lcs_labels.split(",") if label.strip()]
    entity_lcs_skip_labels = [label.strip().upper() for label in args.entity_lcs_skip_labels.split(",") if label.strip()]
    entity_metric = "entity_key_recall" if args.entity_metric == "hard_key_recall" else "entity_lcs"
    model_config: dict[str, Any] = {
        "kind": "gemma4_audio_lora" if args.adapter_path else "gemma4_audio",
        "name": model_name,
        "base_model_path": args.base_model_path,
        "processor_path": args.processor_path or args.base_model_path,
        "adapter_path": args.adapter_path,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "device": args.device,
        "prompt_mode": "fixed" if args.prompt_mode in {"entity_focused", "fixed"} else "dataset",
        "prompt_text": args.prompt_text,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.attn_implementation:
        model_config["attn_implementation"] = args.attn_implementation

    return {
        "dataset": {
            "path": str(args.test_data_path),
            "format": "converted_translation_jsonl",
            "reference_entity_path": str(args.test_entity_path),
            "audio_prefix_from": args.audio_prefix_from,
            "audio_prefix_to": args.audio_prefix_to,
            "limit": args.limit,
        },
        "model": model_config,
        "evaluation": {
            "mode": "generate_and_score",
            "output_dir": str(args.output_dir),
            "metrics": ["bleu", "chrf", entity_metric],
            "entity_lcs_labels": entity_lcs_labels,
            "entity_lcs_skip_labels": entity_lcs_skip_labels,
            "progress_every": args.progress_every,
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sidecar_report = validate_testt_sidecar(args.test_data_path, args.test_entity_path, args.limit)
    (args.output_dir / "sidecar.validation.json").write_text(
        json.dumps(sidecar_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    config = build_eval_config(args)
    config_path = args.config_out or (args.output_dir / "gemma4_base_testt.resolved.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    summary = run_eval_spec(config_path)
    summary["entity_recall"] = summary.get("entity_recall", summary.get("entity_lcs_recall", 0.0))
    summary["metrics_requested"] = ["BLEU", "chrF", "Entity-Recall"]
    summary_path = args.output_dir / "metrics.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sidecar": sidecar_report, "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()

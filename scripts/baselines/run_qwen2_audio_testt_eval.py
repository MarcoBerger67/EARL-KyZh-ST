# -----------------------------------------------------------------------------
# Third-party baseline (best-effort, for paper comparison only).
# This script wires up an external, off-the-shelf model to reproduce one of the
# baseline rows reported in the EARL paper. It depends on third-party model
# weights/APIs that are NOT part of EARL and may break with upstream changes.
# It is not required to train or evaluate EARL itself; the core SFT + GRPO
# pipeline and the entity-recall metric live in the parent scripts/ directory.
# -----------------------------------------------------------------------------
from __future__ import annotations

import os
import sys

# This baseline lives in scripts/baselines/; make the shared ``eval_suite``
# package (in scripts/) importable when the script is run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    "- PER: person names, speakers, authors, historical figures, and transliterated names. Preserve the full name when audible; do not drop given names or family names.\n"
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
    root_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Evaluate testt.jsonl with Qwen2.5-Omni direct speech translation and entity metrics."
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
        "--qwen-model-path",
        type=str,
        default=str(root_dir / "model" / "Qwen2.5-Omni-3B"),
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional Qwen2.5-Omni LoRA adapter path.",
    )
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument(
        "--embedding-model-path",
        type=str,
        default=str(root_dir / "model" / "bert-base-multilingual-cased"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root_dir / "eval_outputs" / "qwen25_omni_3b_testt_lcs",
    )
    parser.add_argument("--audio-prefix-from", type=str, default="")
    parser.add_argument("--audio-prefix-to", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device-map", type=str, default="none")
    parser.add_argument("--entity-soft-tau", type=float, default=0.6)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--config-out", type=Path, default=None)
    parser.add_argument(
        "--entity-metric",
        choices=["lcs", "hard_key_recall"],
        default="hard_key_recall",
        help="Entity-Recall metric. hard_key_recall requires full normalized entity substring match.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["entity_focused", "dataset", "fixed", "template"],
        default="entity_focused",
        help="entity_focused uses the same strict translation prompt family as the current testt evaluation.",
    )
    parser.add_argument("--prompt-text", type=str, default=ENTITY_FOCUSED_TRANSLATION_PROMPT)
    parser.add_argument("--prompt-template", type=str, default="{dataset_prompt}")
    parser.add_argument(
        "--entity-lcs-labels",
        type=str,
        default="LOC,PER,TERM,NUM,ORG,TIME",
        help="Comma-separated entity labels reported by entity_lcs.",
    )
    parser.add_argument(
        "--entity-lcs-skip-labels",
        type=str,
        default="",
        help="Comma-separated labels skipped in overall entity_lcs. Empty string includes all labels.",
    )
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def parse_label_list(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


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
    default_model_name = Path(str(args.qwen_model_path).rstrip("/\\")).name or "Qwen2.5-Omni-3B"
    metrics = ["bleu", "chrf", "entity_key_recall" if args.entity_metric == "hard_key_recall" else "entity_lcs"]
    return {
        "dataset": {
            "path": str(args.test_data_path),
            "format": "converted_translation_jsonl",
            "reference_entity_path": str(args.test_entity_path),
            "audio_prefix_from": args.audio_prefix_from,
            "audio_prefix_to": args.audio_prefix_to,
            "limit": args.limit,
        },
        "model": {
            "kind": "qwen25_omni_s2tt",
            "name": args.model_name or f"{default_model_name}-direct-st",
            "base_model_path": args.qwen_model_path,
            "adapter_path": args.adapter_path,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "torch_dtype": args.torch_dtype,
            "device_map": args.device_map,
            "device": args.device,
            "sampling_rate": args.sampling_rate,
            "prompt_mode": "fixed" if args.prompt_mode in {"entity_focused", "fixed"} else args.prompt_mode,
            "prompt_text": args.prompt_text,
            "prompt_template": args.prompt_template,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "evaluation": {
            "mode": "generate_and_score",
            "output_dir": str(args.output_dir),
            "metrics": metrics,
            "entity_soft_tau": args.entity_soft_tau,
            "entity_lcs_labels": parse_label_list(args.entity_lcs_labels),
            "entity_lcs_skip_labels": parse_label_list(args.entity_lcs_skip_labels),
            "progress_every": args.progress_every,
        },
        "ner": {
            "tokenizer_model": "FINE_ELECTRA_SMALL_ZH",
            "ner_model": "MSRA_NER_ELECTRA_SMALL_ZH",
        },
        "embedding": {
            "model_name": args.embedding_model_path,
            "max_length": 128,
            "pooling": "mean",
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
    config_path = args.config_out or (args.output_dir / "qwen25_omni_3b_testt.resolved.yaml")
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

from __future__ import annotations

"""Entity-injection (prompt-injection) evaluation for Gemma-4 audio LoRA on testt.jsonl.

This script implements the two "entity in the prompt" upper-bound experiments used as
contrast points for entity-aware group-relative reinforcement:

  1. --entity-source gold            (验证侧实体注入 / per-sample oracle skyline)
       Inject every gold entity of the *current test sample* (from the gold NER sidecar)
       into the prompt and ask the model to render each one. This is the tightest
       upper bound: the model is told exactly which target-side entities must appear.
       It is a *skyline / diagnostic*, NOT a comparable baseline, because it uses
       test-set answers at inference time.

  2. --entity-source train-glossary  (训练侧词表注入 / realistic glossary upper bound)
       Build an entity gazetteer offline from a *training* NER sidecar, then for each
       test sample inject only the gold entities whose text is covered by that training
       gazetteer. This represents "what a domain glossary built from training data would
       have known" — a leakage-controlled, realistic upper bound on glossary / terminology
       injection methods (cf. WMT24 terminology integration). Entities present in the
       reference but absent from the training glossary are never injected (a glossary miss).

In both cases scoring is always done against the gold sidecar (reference_entity_path),
so BLEU / chrF / entity_lcs are computed exactly like every other eval in this repo and
the numbers are directly comparable with the no-injection baselines.

Mechanism: we do NOT modify eval_suite. We materialize a derived test JSONL in which each
user message text is replaced by `base_prompt + entity_hint_block`, then run the standard
eval with prompt_mode="dataset".
"""

import argparse
import copy
import json
from pathlib import Path
from typing import Any


# Some testt JSONL files are exported on Windows with raw, unescaped single backslashes in
# audio paths, which is invalid JSON. In these
# exports EVERY backslash is a literal path separator that should have been "\\". We cannot
# trust a "fix only invalid escapes" rule because "\t", "\n", etc. would silently corrupt
# paths like "...\testt..." into a TAB. So on strict-parse failure we double *all*
# backslashes and retry. Strictly valid lines never reach the fallback, so this is safe.
def _loads_lenient(line: str) -> dict[str, Any]:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return json.loads(line.replace("\\", "\\\\"))

import yaml

from eval_suite.runner import run_eval_spec


# Base translation prompt — kept identical to the GRPO runtime prompt so the only
# difference vs. the no-injection eval is the appended entity hint block.
BASE_TRANSLATION_PROMPT = (
    "You are a strict Kyrgyz-to-Chinese speech translation system.\n\n"
    "Translate the input speech segment into fluent Simplified Chinese.\n\n"
    "Rules:\n"
    "1. Output only the final translation.\n"
    "2. Output one single line only.\n"
    "3. Output the complete translation of the entire speech segment.\n"
    "4. Do not omit any translated content.\n"
    "5. Do not stop after translating only the beginning of the sentence.\n"
    "6. Output Simplified Chinese only. Do not output Kyrgyz, English, "
    "explanations, notes, labels, speaker tags, or markdown.\n"
    '7. Do not output prefixes or suffixes such as "Key words:", "Line:", '
    '"Translation:", "Answer:", or quotation marks around the answer.\n'
    "8. Preserve the meaning faithfully. Do not summarize, rewrite, expand, "
    "or invent content not supported by the speech.\n"
    "9. Preserve names, places, organizations, and numbers as accurately as possible.\n"
    "10. Write numbers using Arabic numerals, e.g. 1.7, 3, 80%.\n"
    "11. If some part is unclear, translate conservatively based on the audio and "
    "context, and do not hallucinate."
)

VALID_LABELS = {"PER", "LOC", "ORG", "TIME", "NUM", "TERM", "TITLE"}


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
            data_ids.append(record_id(_loads_lenient(line), f"line_{line_number}"))
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
            record = _loads_lenient(line)
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
        "json_load_mode": "lenient_windows_backslash_fallback",
    }


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Entity prompt-injection eval for Gemma-4 audio LoRA. "
            "Use --entity-source gold for the per-sample oracle skyline, or "
            "--entity-source train-glossary for the training-glossary upper bound."
        )
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
        help="Gold NER sidecar for the test set. Used both for injection (gold lane) and for scoring (always).",
    )
    parser.add_argument(
        "--train-entity-path",
        type=Path,
        default=None,
        help="Training NER sidecar used to build the gazetteer for --entity-source train-glossary.",
    )
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--processor-path", type=str, default=None)
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional LoRA adapter path. Omit this for a pure Gemma-4 base-model oracle injection run.",
    )
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-prefix-from", type=str, default="")
    parser.add_argument("--audio-prefix-to", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device-map", type=str, default="none")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=20)

    # ── Injection controls ───────────────────────────────────────────────────
    parser.add_argument(
        "--entity-source",
        choices=["gold", "train-glossary"],
        default="gold",
        help="gold = per-sample oracle skyline; train-glossary = training-derived glossary upper bound.",
    )
    parser.add_argument(
        "--inject-format",
        choices=["full", "type_only"],
        default="full",
        help="full = list entity strings + labels; type_only = reveal only label counts (weak oracle).",
    )
    parser.add_argument(
        "--inject-labels",
        type=str,
        default="LOC,PER,ORG,TERM,NUM,TIME,TITLE",
        help="Comma-separated labels eligible for injection.",
    )
    parser.add_argument(
        "--inject-skip-labels",
        type=str,
        default="",
        help="Comma-separated labels excluded from injection (e.g. PER for transliterated names).",
    )

    # ── Metric controls (kept identical to the other testt eval scripts) ──────
    parser.add_argument("--entity-lcs-labels", type=str, default="LOC,PER,TERM,NUM,ORG,TIME")
    parser.add_argument(
        "--entity-lcs-skip-labels",
        type=str,
        default="",
        help=(
            "Labels skipped in the aggregate entity recall. Empty string includes all labels. "
            "This also applies when --entity-metric hard_key_recall."
        ),
    )
    parser.add_argument(
        "--entity-metric",
        choices=["lcs", "hard_key_recall"],
        default="hard_key_recall",
        help="Entity metric. hard_key_recall requires the full normalized reference entity to appear in prediction.",
    )
    parser.add_argument("--config-out", type=Path, default=None)
    return parser.parse_args()


def _csv_labels(value: str) -> set[str]:
    return {item.strip().upper() for item in value.split(",") if item.strip()}


def _load_entity_sidecar(path: Path) -> dict[str, list[dict[str, str]]]:
    sidecar: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = _loads_lenient(line)
            sample_id = str(record.get("id") or record.get("key") or f"line_{line_number}")
            entities: list[dict[str, str]] = []
            seen: set[tuple[str, str]] = set()
            for item in record.get("entities", []):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "") or "").strip()
                label = str(item.get("label", "") or "").strip().upper()
                if not text or label not in VALID_LABELS:
                    continue
                key = (text, label)
                if key in seen:
                    continue
                seen.add(key)
                entities.append({"text": text, "label": label})
            sidecar[sample_id] = entities
    return sidecar


def _build_train_gazetteer(path: Path) -> set[str]:
    gazetteer: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = _loads_lenient(line)
            for item in record.get("entities", []):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "") or "").strip()
                label = str(item.get("label", "") or "").strip().upper()
                if text and label in VALID_LABELS:
                    gazetteer.add(text)
    return gazetteer


def _select_injected_entities(
    gold_entities: list[dict[str, str]],
    entity_source: str,
    gazetteer: set[str] | None,
    inject_labels: set[str],
    inject_skip_labels: set[str],
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for entity in gold_entities:
        label = entity["label"]
        if label not in inject_labels or label in inject_skip_labels:
            continue
        if entity_source == "train-glossary":
            if gazetteer is None or entity["text"] not in gazetteer:
                continue
        selected.append(entity)
    return selected


def _format_entity_block(entities: list[dict[str, str]], inject_format: str) -> str:
    if not entities:
        return ""
    if inject_format == "type_only":
        counts: dict[str, int] = {}
        for entity in entities:
            counts[entity["label"]] = counts.get(entity["label"], 0) + 1
        summary = ", ".join(f"{label} x{count}" for label, count in sorted(counts.items()))
        return (
            "\n\nThe speech contains entities of these types (with counts): "
            f"{summary}. Translate every such entity faithfully and completely; do not drop or merge them."
        )
    lines = "\n".join(f"- {entity['text']} [{entity['label']}]" for entity in entities)
    return (
        "\n\nThe speech mentions the following entities. Render EACH of them in your Chinese "
        "translation exactly as written below, without omission, substitution, or reordering of the entity itself:\n"
        f"{lines}"
    )


def _rewrite_user_text(record: dict[str, Any], new_text: str) -> dict[str, Any]:
    new_record = copy.deepcopy(record)
    messages = new_record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"Record {new_record.get('id')!r} has no messages list.")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        kept = [item for item in content if not (isinstance(item, dict) and item.get("type") == "text")]
        kept.append({"type": "text", "text": new_text})
        message["content"] = kept
        return new_record
    raise ValueError(f"Record {new_record.get('id')!r} has no user message to inject into.")


def build_derived_dataset(args: argparse.Namespace) -> dict[str, Any]:
    inject_labels = _csv_labels(args.inject_labels)
    inject_skip_labels = _csv_labels(args.inject_skip_labels)
    gold_sidecar = _load_entity_sidecar(args.test_entity_path)

    gazetteer: set[str] | None = None
    if args.entity_source == "train-glossary":
        if args.train_entity_path is None:
            raise ValueError("--train-entity-path is required for --entity-source train-glossary.")
        gazetteer = _build_train_gazetteer(args.train_entity_path)

    derived_path = args.output_dir / "derived_testt.inject.jsonl"
    derived_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    samples_with_hints = 0
    injected_entity_total = 0
    eligible_gold_total = 0
    injected_by_label: dict[str, int] = {}
    with args.test_data_path.open("r", encoding="utf-8") as src, derived_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            record = _loads_lenient(line)
            sample_id = str(record.get("id") or record.get("key") or f"line_{line_number}")
            gold_entities = gold_sidecar.get(sample_id, [])
            eligible_gold_total += sum(
                1
                for entity in gold_entities
                if entity["label"] in inject_labels and entity["label"] not in inject_skip_labels
            )
            injected = _select_injected_entities(
                gold_entities, args.entity_source, gazetteer, inject_labels, inject_skip_labels
            )
            entity_block = _format_entity_block(injected, args.inject_format)
            new_text = BASE_TRANSLATION_PROMPT + entity_block
            dst.write(json.dumps(_rewrite_user_text(record, new_text), ensure_ascii=False) + "\n")

            rows += 1
            if injected:
                samples_with_hints += 1
            injected_entity_total += len(injected)
            for entity in injected:
                injected_by_label[entity["label"]] = injected_by_label.get(entity["label"], 0) + 1
            if args.limit is not None and rows >= args.limit:
                break

    manifest = {
        "entity_source": args.entity_source,
        "inject_format": args.inject_format,
        "inject_labels": sorted(inject_labels),
        "inject_skip_labels": sorted(inject_skip_labels),
        "derived_dataset_path": str(derived_path),
        "rows": rows,
        "samples_with_hints": samples_with_hints,
        "samples_with_hints_ratio": (samples_with_hints / rows) if rows else 0.0,
        "injected_entity_total": injected_entity_total,
        "injected_by_label": dict(sorted(injected_by_label.items())),
        "eligible_gold_entity_total": eligible_gold_total,
        "glossary_coverage_over_eligible_gold": (
            injected_entity_total / eligible_gold_total if eligible_gold_total else None
        ),
        "train_gazetteer_size": (len(gazetteer) if gazetteer is not None else None),
        "note": (
            "Scoring uses the gold sidecar (test_entity_path); injection only alters the prompt. "
            "gold = per-sample oracle skyline; train-glossary = realistic glossary upper bound."
        ),
    }
    return manifest


def build_eval_config(args: argparse.Namespace, derived_path: Path) -> dict[str, Any]:
    entity_lcs_labels = [label.strip().upper() for label in args.entity_lcs_labels.split(",") if label.strip()]
    entity_lcs_skip_labels = [label.strip().upper() for label in args.entity_lcs_skip_labels.split(",") if label.strip()]
    entity_metric = "entity_key_recall" if args.entity_metric == "hard_key_recall" else "entity_lcs"
    base_name = Path(args.base_model_path.rstrip("/")).name
    model_name = args.model_name or f"{base_name}-inject-{args.entity_source}-{args.inject_format}"
    model_config: dict[str, Any] = {
        "kind": "gemma4_audio_lora" if args.adapter_path else "gemma4_audio",
        "name": model_name,
        "base_model_path": args.base_model_path,
        "processor_path": args.processor_path or args.base_model_path,
        "adapter_path": args.adapter_path,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "torch_dtype": args.torch_dtype,
        "device": args.device,
        "device_map": args.device_map,
        "sampling_rate": args.sampling_rate,
        # Per-sample injected prompt is materialized in the derived dataset's user text.
        "prompt_mode": "dataset",
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 1.0,
    }
    if args.attn_implementation:
        model_config["attn_implementation"] = args.attn_implementation
    return {
        "dataset": {
            "path": str(derived_path.resolve()),
            "format": "converted_translation_jsonl",
            # Always score against the GOLD sidecar, never the injected subset.
            "reference_entity_path": str(args.test_entity_path.resolve()),
            "audio_prefix_from": args.audio_prefix_from,
            "audio_prefix_to": args.audio_prefix_to,
            "limit": args.limit,
        },
        "model": model_config,
        "evaluation": {
            "mode": "generate_and_score",
            "output_dir": str(args.output_dir.resolve()),
            "metrics": ["bleu", "chrf", entity_metric],
            "entity_lcs_labels": entity_lcs_labels,
            "entity_lcs_skip_labels": entity_lcs_skip_labels,
            "progress_every": args.progress_every,
        },
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Validate alignment of test data and gold sidecar (same check as the other testt evals).
    sidecar_report = validate_testt_sidecar(args.test_data_path, args.test_entity_path, args.limit)
    (args.output_dir / "sidecar.validation.json").write_text(
        json.dumps(sidecar_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = build_derived_dataset(args)
    (args.output_dir / "injection.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"injection_manifest": manifest}, ensure_ascii=False), flush=True)

    config = build_eval_config(args, Path(manifest["derived_dataset_path"]))
    config_path = args.config_out or (args.output_dir / "gemma4_entity_inject_eval.resolved.yaml")
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")

    summary = run_eval_spec(config_path)
    summary["entity_recall"] = summary.get(
        "entity_key_recall",
        summary.get("entity_recall", summary.get("entity_lcs_recall", 0.0)),
    )
    summary["entity_metric"] = args.entity_metric
    summary["metrics_requested"] = ["BLEU", "chrF", "Entity-Recall"]
    summary["prompt_variant"] = f"entity_inject_{args.entity_source}_{args.inject_format}"
    summary["injection_manifest"] = manifest
    summary_path = args.output_dir / "metrics.entity_inject.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

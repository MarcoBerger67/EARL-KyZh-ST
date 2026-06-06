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

import torch
import yaml

from eval_suite.config import load_eval_spec
from eval_suite.data import load_samples
from eval_suite.runner import run_eval_spec


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Evaluate MADLAD400 MT LoRA on testt references with BLEU/chrF/entity_lcs."
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
        "--source-prediction-path",
        type=Path,
        required=True,
        help="JSONL providing Kyrgyz source text, usually ASR predictions with intermediate_asr_text.",
    )
    parser.add_argument(
        "--source-text-fields",
        type=str,
        default="intermediate_asr_text,prediction_text,prediction,source_text,text",
        help="Comma-separated fields to try in --source-prediction-path.",
    )
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root_dir / "eval_outputs" / "madlad400_3b_mt_lora_testt_lcs",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--target-prefix", type=str, default="<2zh>")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device-map", type=str, default="none")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--entity-lcs-labels", type=str, default="LOC,PER,TERM,NUM,ORG,TIME")
    parser.add_argument(
        "--entity-lcs-skip-labels",
        type=str,
        default="",
        help='Comma-separated labels skipped in overall entity_lcs. Default "" includes PER.',
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_source_texts(path: Path, fields: list[str]) -> dict[str, str]:
    sources: dict[str, str] = {}
    for index, row in enumerate(iter_jsonl(path), start=1):
        sample_id = str(row.get("id") or row.get("sample_id") or row.get("key") or f"line_{index}")
        for field in fields:
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                sources[sample_id] = value.strip()
                break
    return sources


def load_existing_predictions(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(iter_jsonl(path), start=1):
        sample_id = str(row.get("id") or row.get("key") or f"line_{index}")
        prediction = str(row.get("prediction_text", row.get("prediction", "")) or "").strip()
        if prediction:
            rows[sample_id] = row
    return rows


def resolve_torch_dtype(value: str) -> torch.dtype | None:
    if value == "auto":
        return None
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[value]


def write_eval_config(args: argparse.Namespace, predictions_path: Path, config_path: Path) -> None:
    entity_lcs_labels = [label.strip().upper() for label in args.entity_lcs_labels.split(",") if label.strip()]
    entity_lcs_skip_labels = [label.strip().upper() for label in args.entity_lcs_skip_labels.split(",") if label.strip()]
    config = {
        "dataset": {
            "path": str(args.test_data_path),
            "format": "converted_translation_jsonl",
            "reference_entity_path": str(args.test_entity_path),
            "limit": args.limit,
        },
        "model": {
            "kind": "offline_predictions",
            "name": args.model_name or "madlad400-3b-mt-lora",
        },
        "evaluation": {
            "mode": "offline_predictions",
            "output_dir": str(args.output_dir),
            "prediction_path": str(predictions_path),
            "prediction_id_field": "id",
            "prediction_text_field": "prediction_text",
            "metrics": ["bleu", "chrf", "entity_lcs"],
            "entity_lcs_labels": entity_lcs_labels,
            "entity_lcs_skip_labels": entity_lcs_skip_labels,
            "progress_every": args.progress_every,
        },
    }
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.madlad400_mt.jsonl"
    errors_path = args.output_dir / "errors.madlad400_mt.jsonl"
    resolved_config_path = args.output_dir / "madlad400_mt_eval.resolved.yaml"
    write_eval_config(args, predictions_path, resolved_config_path)

    fields = [field.strip() for field in args.source_text_fields.split(",") if field.strip()]
    source_texts = load_source_texts(args.source_prediction_path, fields)

    spec = load_eval_spec(resolved_config_path)
    samples = load_samples(spec.dataset)
    existing = load_existing_predictions(predictions_path) if args.resume else {}
    pending_samples = [sample for sample in samples if sample.sample_id not in existing]

    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    model_kwargs: dict[str, Any] = {}
    dtype = resolve_torch_dtype(args.torch_dtype)
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model_path, **model_kwargs)
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    if args.device_map.lower() == "none":
        model.to(args.device)
    model.eval()

    def translate_batch(texts: list[str]) -> list[str]:
        prefixed = [f"{args.target_prefix} {text}".strip() for text in texts]
        inputs = tokenizer(prefixed, return_tensors="pt", padding=True, truncation=True)
        if args.device_map.lower() == "none":
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)

    done = len(existing)
    with predictions_path.open("a", encoding="utf-8") as out, errors_path.open("a", encoding="utf-8") as err_out:
        for start in range(0, len(pending_samples), args.batch_size):
            batch = pending_samples[start : start + args.batch_size]
            batch_sources: list[str] = []
            ok_samples = []
            for sample in batch:
                source_text = source_texts.get(sample.sample_id, sample.source_text).strip()
                if not source_text:
                    err_out.write(
                        json.dumps(
                            {
                                "id": sample.sample_id,
                                "error": "Missing source text. Provide --source-prediction-path with intermediate_asr_text or source_text.",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    err_out.flush()
                    done += 1
                    continue
                batch_sources.append(source_text)
                ok_samples.append(sample)
            if ok_samples:
                try:
                    translations = translate_batch(batch_sources)
                except Exception as exc:
                    for sample in ok_samples:
                        err_out.write(json.dumps({"id": sample.sample_id, "error": str(exc)}, ensure_ascii=False) + "\n")
                    err_out.flush()
                    done += len(ok_samples)
                else:
                    for sample, source_text, prediction in zip(ok_samples, batch_sources, translations):
                        out.write(
                            json.dumps(
                                {
                                    "id": sample.sample_id,
                                    "prediction_text": prediction.strip(),
                                    "reference_text": sample.reference_text,
                                    "source_text": source_text,
                                    "model_name": args.model_name or "madlad400-3b-mt-lora",
                                    "adapter_path": args.adapter_path,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    out.flush()
                    done += len(ok_samples)
            if args.progress_every > 0 and done % args.progress_every == 0:
                print(json.dumps({"event": "madlad400_mt_progress", "done": done, "total": len(samples)}, ensure_ascii=False), flush=True)

    prediction_ids = set(load_existing_predictions(predictions_path).keys())
    expected_ids = {sample.sample_id for sample in samples}
    missing_ids = sorted(expected_ids - prediction_ids)
    if missing_ids:
        missing_path = args.output_dir / "missing_predictions.json"
        missing_path.write_text(
            json.dumps(
                {
                    "num_expected": len(expected_ids),
                    "num_predictions": len(prediction_ids),
                    "num_missing": len(missing_ids),
                    "missing_ids": missing_ids[:1000],
                    "hint": "Fix missing source texts or model errors and rerun with --resume.",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "event": "skip_metrics_missing_predictions",
                    "num_expected": len(expected_ids),
                    "num_predictions": len(prediction_ids),
                    "num_missing": len(missing_ids),
                    "missing_path": str(missing_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return

    summary = run_eval_spec(resolved_config_path)
    summary["entity_recall"] = summary.get("entity_recall", summary.get("entity_lcs_recall", 0.0))
    summary["metrics_requested"] = ["BLEU", "chrF", "Entity-Recall"]
    summary["source_prediction_path"] = str(args.source_prediction_path)
    summary_path = args.output_dir / "metrics.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

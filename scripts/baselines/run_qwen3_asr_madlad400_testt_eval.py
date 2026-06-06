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
from eval_suite.model_adapters import (
    _patch_qwen3_asr_config_compat,
    _patch_qwen3_asr_generation_compat,
    _patch_qwen3_asr_rope_compat,
    _patch_transformers_check_model_inputs_compat,
    _resolve_torch_dtype,
)
from eval_suite.runner import run_eval_spec


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-ASR-1.7B + MADLAD400 MT LoRA on testt with BLEU/chrF/entity_lcs."
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
    parser.add_argument("--qwen3-asr-model-path", type=str, required=True)
    parser.add_argument("--madlad-model-path", type=str, required=True)
    parser.add_argument(
        "--madlad-adapter-path",
        type=str,
        default="",
        help="Optional MADLAD400 MT LoRA adapter path. Empty means evaluate the base MT model.",
    )
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root_dir / "eval_outputs" / "qwen3_asr_madlad400_testt_lcs",
    )
    parser.add_argument("--audio-prefix-from", type=str, default="")
    parser.add_argument("--audio-prefix-to", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=16, help="MADLAD MT batch size.")
    parser.add_argument("--asr-batch-size", type=int, default=1, help="Qwen3-ASR batch size. Keep 1 if the runtime is fragile.")
    parser.add_argument("--asr-max-new-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--target-prefix", type=str, default="<2zh>")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--mt-torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--mt-device-map", type=str, default="none")
    parser.add_argument("--asr-attn-implementation", type=str, default="eager")
    parser.add_argument("--asr-language", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument(
        "--entity-metric",
        choices=["lcs", "hard_key_recall"],
        default="hard_key_recall",
        help="Entity-Recall metric. hard_key_recall requires exact normalized entity substring match.",
    )
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


def parse_label_list(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def write_eval_config(args: argparse.Namespace, predictions_path: Path, config_path: Path) -> None:
    entity_metric = "entity_key_recall" if args.entity_metric == "hard_key_recall" else "entity_lcs"
    config = {
        "dataset": {
            "path": str(args.test_data_path),
            "format": "converted_translation_jsonl",
            "reference_entity_path": str(args.test_entity_path),
            "audio_prefix_from": args.audio_prefix_from,
            "audio_prefix_to": args.audio_prefix_to,
            "limit": args.limit,
        },
        "model": {
            "kind": "offline_predictions",
            "name": args.model_name or "qwen3-asr-1.7b__madlad400-3b-mt-lora",
        },
        "evaluation": {
            "mode": "offline_predictions",
            "output_dir": str(args.output_dir),
            "prediction_path": str(predictions_path),
            "prediction_id_field": "id",
            "prediction_text_field": "prediction_text",
            "metrics": ["bleu", "chrf", entity_metric],
            "entity_lcs_labels": parse_label_list(args.entity_lcs_labels),
            "entity_lcs_skip_labels": parse_label_list(args.entity_lcs_skip_labels),
            "progress_every": args.progress_every,
        },
    }
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")


def load_qwen3_asr(args: argparse.Namespace) -> Any:
    _patch_transformers_check_model_inputs_compat()
    from qwen_asr import Qwen3ASRModel

    _patch_qwen3_asr_config_compat()
    _patch_qwen3_asr_rope_compat()
    _patch_qwen3_asr_generation_compat()

    model_kwargs: dict[str, Any] = {
        "dtype": _resolve_torch_dtype(args.torch_dtype),
        "device_map": args.device_map,
        "max_inference_batch_size": args.asr_batch_size,
        "max_new_tokens": args.asr_max_new_tokens,
    }
    if args.asr_attn_implementation:
        model_kwargs["attn_implementation"] = args.asr_attn_implementation
    try:
        return Qwen3ASRModel.from_pretrained(args.qwen3_asr_model_path, **model_kwargs)
    except TypeError:
        model_kwargs.pop("max_inference_batch_size", None)
        model_kwargs.pop("max_new_tokens", None)
        return Qwen3ASRModel.from_pretrained(args.qwen3_asr_model_path, **model_kwargs)


def transcribe_batch(asr_model: Any, audio_paths: list[str], language: str | None) -> list[str]:
    result = asr_model.transcribe(audio=audio_paths, language=language)
    if isinstance(result, str):
        return [result]
    texts: list[str] = []
    for item in result:
        if isinstance(item, str):
            texts.append(item.strip())
        elif isinstance(item, dict):
            texts.append(str(item.get("text", item.get("transcript", ""))).strip())
        else:
            texts.append(str(getattr(item, "text", item)).strip())
    return texts


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.qwen3_asr_madlad400.jsonl"
    errors_path = args.output_dir / "errors.qwen3_asr_madlad400.jsonl"
    resolved_config_path = args.output_dir / "qwen3_asr_madlad400_eval.resolved.yaml"
    write_eval_config(args, predictions_path, resolved_config_path)

    spec = load_eval_spec(resolved_config_path)
    samples = load_samples(spec.dataset)
    existing = load_existing_predictions(predictions_path) if args.resume else {}
    pending_samples = [sample for sample in samples if sample.sample_id not in existing]

    asr_model = load_qwen3_asr(args)

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.madlad_model_path)
    mt_kwargs: dict[str, Any] = {}
    mt_dtype = _resolve_torch_dtype(args.mt_torch_dtype)
    if mt_dtype != "auto":
        mt_kwargs["torch_dtype"] = mt_dtype
    if args.mt_device_map.lower() != "none":
        mt_kwargs["device_map"] = args.mt_device_map
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(args.madlad_model_path, **mt_kwargs)
    if args.madlad_adapter_path:
        from peft import PeftModel

        mt_model = PeftModel.from_pretrained(mt_model, args.madlad_adapter_path, is_trainable=False)
    if args.mt_device_map.lower() == "none":
        mt_model.to(args.device)
    mt_model.eval()

    def translate_batch(texts: list[str]) -> list[str]:
        prefixed = [f"{args.target_prefix} {text}".strip() for text in texts]
        inputs = tokenizer(prefixed, return_tensors="pt", padding=True, truncation=True)
        if args.mt_device_map.lower() == "none":
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
        with torch.inference_mode():
            generated = mt_model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)

    done = len(existing)
    with predictions_path.open("a", encoding="utf-8") as out, errors_path.open("a", encoding="utf-8") as err_out:
        for start in range(0, len(pending_samples), args.batch_size):
            mt_batch_samples = pending_samples[start : start + args.batch_size]
            asr_texts_by_id: dict[str, str] = {}
            for asr_start in range(0, len(mt_batch_samples), args.asr_batch_size):
                asr_samples = mt_batch_samples[asr_start : asr_start + args.asr_batch_size]
                audio_paths = []
                valid_asr_samples = []
                for sample in asr_samples:
                    if not sample.audio_path:
                        err_out.write(json.dumps({"id": sample.sample_id, "error": "Missing audio path."}, ensure_ascii=False) + "\n")
                        err_out.flush()
                        done += 1
                        continue
                    audio_paths.append(sample.audio_path)
                    valid_asr_samples.append(sample)
                if not valid_asr_samples:
                    continue
                try:
                    asr_texts = transcribe_batch(asr_model, audio_paths, args.asr_language)
                except Exception as exc:
                    for sample in valid_asr_samples:
                        err_out.write(json.dumps({"id": sample.sample_id, "audio_path": sample.audio_path, "error": f"ASR failed: {exc}"}, ensure_ascii=False) + "\n")
                    err_out.flush()
                    done += len(valid_asr_samples)
                    continue
                for sample, asr_text in zip(valid_asr_samples, asr_texts):
                    if asr_text.strip():
                        asr_texts_by_id[sample.sample_id] = asr_text.strip()
                    else:
                        err_out.write(json.dumps({"id": sample.sample_id, "audio_path": sample.audio_path, "error": "ASR returned empty text."}, ensure_ascii=False) + "\n")
                        err_out.flush()
                        done += 1

            ok_samples = [sample for sample in mt_batch_samples if sample.sample_id in asr_texts_by_id]
            if ok_samples:
                source_texts = [asr_texts_by_id[sample.sample_id] for sample in ok_samples]
                try:
                    translations = translate_batch(source_texts)
                except Exception as exc:
                    for sample in ok_samples:
                        err_out.write(json.dumps({"id": sample.sample_id, "audio_path": sample.audio_path, "error": f"MT failed: {exc}"}, ensure_ascii=False) + "\n")
                    err_out.flush()
                    done += len(ok_samples)
                else:
                    for sample, asr_text, translation in zip(ok_samples, source_texts, translations):
                        out.write(
                            json.dumps(
                                {
                                    "id": sample.sample_id,
                                    "prediction_text": translation.strip(),
                                    "reference_text": sample.reference_text,
                                    "audio_path": sample.audio_path,
                                    "intermediate_asr_text": asr_text,
                                    "source_text": asr_text,
                                    "model_name": args.model_name or "qwen3-asr-1.7b__madlad400-3b-mt-lora",
                                    "adapter_path": args.madlad_adapter_path or None,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    out.flush()
                    done += len(ok_samples)

            if args.progress_every > 0 and done % args.progress_every == 0:
                print(json.dumps({"event": "qwen3_asr_madlad400_progress", "done": done, "total": len(samples)}, ensure_ascii=False), flush=True)

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
                    "hint": "Fix ASR/MT errors and rerun with --resume.",
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
    summary_path = args.output_dir / "metrics.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

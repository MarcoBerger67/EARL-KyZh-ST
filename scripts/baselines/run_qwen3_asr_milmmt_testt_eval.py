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


DEFAULT_PROMPT_TEMPLATE = (
    "Translate the following Kyrgyz text into Simplified Chinese.\n"
    "Only output the translation, with no explanations, labels, markdown, or newlines.\n\n"
    "{source_text}"
)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen3-ASR-1.7B + MiLMMT-46-4B LoRA on testt with BLEU/chrF/entity_lcs."
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
    parser.add_argument("--milmmt-model-path", type=str, required=True)
    parser.add_argument("--milmmt-adapter-path", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root_dir / "eval_outputs" / "qwen3_asr_milmmt_testt_lcs",
    )
    parser.add_argument("--audio-prefix-from", type=str, default="")
    parser.add_argument("--audio-prefix-to", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=8, help="MiLMMT MT batch size.")
    parser.add_argument("--asr-batch-size", type=int, default=1, help="Qwen3-ASR batch size.")
    parser.add_argument("--asr-max-new-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument(
        "--decode-full-output-if-empty",
        action="store_true",
        default=True,
        help="Fallback for models whose generate() already returns continuation-only ids.",
    )
    parser.add_argument("--prompt-template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
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
            "name": args.model_name or "qwen3-asr-1.7b__milmmt-46-4b-lora",
        },
        "evaluation": {
            "mode": "offline_predictions",
            "output_dir": str(args.output_dir),
            "prediction_path": str(predictions_path),
            "prediction_id_field": "id",
            "prediction_text_field": "prediction_text",
            "metrics": ["bleu", "chrf", "entity_lcs"],
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


def build_prompt(source_text: str, template: str) -> str:
    return template.format(source_text=source_text)


def build_messages(source_text: str, template: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": build_prompt(source_text, template)}]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.qwen3_asr_milmmt.jsonl"
    errors_path = args.output_dir / "errors.qwen3_asr_milmmt.jsonl"
    resolved_config_path = args.output_dir / "qwen3_asr_milmmt_eval.resolved.yaml"
    write_eval_config(args, predictions_path, resolved_config_path)

    spec = load_eval_spec(resolved_config_path)
    samples = load_samples(spec.dataset)
    existing = load_existing_predictions(predictions_path) if args.resume else {}
    pending_samples = [sample for sample in samples if sample.sample_id not in existing]

    asr_model = load_qwen3_asr(args)

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.milmmt_model_path)
    tokenizer = getattr(processor, "tokenizer", processor)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    mt_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
    mt_dtype = _resolve_torch_dtype(args.mt_torch_dtype)
    if mt_dtype != "auto":
        mt_kwargs["torch_dtype"] = mt_dtype
    else:
        mt_kwargs["torch_dtype"] = "auto"
    if args.mt_device_map.lower() != "none":
        mt_kwargs["device_map"] = args.mt_device_map
    mt_model = AutoModelForImageTextToText.from_pretrained(args.milmmt_model_path, **mt_kwargs)
    mt_model = PeftModel.from_pretrained(mt_model, args.milmmt_adapter_path, is_trainable=False)
    if getattr(mt_model.generation_config, "pad_token_id", None) is None:
        mt_model.generation_config.pad_token_id = tokenizer.pad_token_id
    if getattr(mt_model.generation_config, "eos_token_id", None) is None and tokenizer.eos_token_id is not None:
        mt_model.generation_config.eos_token_id = tokenizer.eos_token_id
    if hasattr(mt_model, "config"):
        mt_model.config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            mt_model.config.eos_token_id = tokenizer.eos_token_id
    if args.mt_device_map.lower() == "none":
        mt_model.to(args.device)
    mt_model.eval()

    def translate_batch(texts: list[str]) -> list[str]:
        prompts = [build_messages(text, args.prompt_template) for text in texts]
        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            encoded = tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
            )
            if isinstance(encoded, dict):
                inputs = encoded
            else:
                inputs = {"input_ids": encoded}
        else:
            prompt_texts = [build_prompt(text, args.prompt_template) for text in texts]
            inputs = tokenizer(
                prompt_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
            )
        if args.mt_device_map.lower() == "none":
            inputs = {key: value.to(args.device) for key, value in inputs.items()}
        input_lengths = inputs["input_ids"].shape[-1]
        with torch.inference_mode():
            generated = mt_model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
        continuations = generated[:, input_lengths:]
        decoded = tokenizer.batch_decode(continuations, skip_special_tokens=True)
        if args.decode_full_output_if_empty and all(not item.strip() for item in decoded):
            full_decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            prompt_texts = [build_prompt(text, args.prompt_template) for text in texts]
            cleaned: list[str] = []
            for full_text, prompt_text in zip(full_decoded, prompt_texts):
                value = full_text.strip()
                if prompt_text and value.startswith(prompt_text):
                    value = value[len(prompt_text) :].strip()
                for marker in ("Assistant:", "assistant:", "模型:", "答：", "答案："):
                    if marker in value:
                        value = value.split(marker, 1)[-1].strip()
                cleaned.append(value)
            decoded = cleaned
        return decoded

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
                    if len(valid_asr_samples) <= 1:
                        for sample in valid_asr_samples:
                            err_out.write(json.dumps({"id": sample.sample_id, "audio_path": sample.audio_path, "error": f"ASR failed: {exc}"}, ensure_ascii=False) + "\n")
                        err_out.flush()
                        done += len(valid_asr_samples)
                        continue
                    err_out.write(
                        json.dumps(
                            {
                                "ids": [sample.sample_id for sample in valid_asr_samples],
                                "error": f"ASR batch failed, retrying one by one: {exc}",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    err_out.flush()
                    for sample in valid_asr_samples:
                        try:
                            single_texts = transcribe_batch(asr_model, [sample.audio_path], args.asr_language)
                        except Exception as single_exc:
                            err_out.write(
                                json.dumps(
                                    {
                                        "id": sample.sample_id,
                                        "audio_path": sample.audio_path,
                                        "error": f"ASR failed after single-sample retry: {single_exc}",
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            err_out.flush()
                            done += 1
                            continue
                        single_text = single_texts[0].strip() if single_texts else ""
                        if single_text:
                            asr_texts_by_id[sample.sample_id] = single_text
                        else:
                            err_out.write(
                                json.dumps(
                                    {"id": sample.sample_id, "audio_path": sample.audio_path, "error": "ASR returned empty text after single-sample retry."},
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            err_out.flush()
                            done += 1
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
                                    "model_name": args.model_name or "qwen3-asr-1.7b__milmmt-46-4b-lora",
                                    "adapter_path": args.milmmt_adapter_path,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    out.flush()
                    done += len(ok_samples)

            if args.progress_every > 0 and done % args.progress_every == 0:
                print(json.dumps({"event": "qwen3_asr_milmmt_progress", "done": done, "total": len(samples)}, ensure_ascii=False), flush=True)

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

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from eval_suite.config import load_eval_spec
from eval_suite.data import load_samples
from eval_suite.model_adapters import (
    _get_pad_token_id,
    _load_processor_by_declared_class,
    _move_to_device,
    _resolve_torch_dtype,
)
from eval_suite.runner import run_eval_spec
from eval_suite.text_metrics import get_sacrebleu
from run_gemma4_base_testt_eval import ENTITY_FOCUSED_TRANSLATION_PROMPT


VALID_LABELS = {"PER", "LOC", "ORG", "TIME", "NUM", "TERM", "TITLE"}


def _loads_lenient(line: str) -> dict[str, Any]:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return json.loads(line.replace("\\", "\\\\"))


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


def materialize_lenient_test_jsonl(data_path: Path, output_dir: Path, limit: int | None) -> Path:
    derived_path = output_dir / "derived_testt.mbr_input.jsonl"
    rows = 0
    with data_path.open("r", encoding="utf-8") as src, derived_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as dst:
        for line in src:
            if not line.strip():
                continue
            dst.write(json.dumps(_loads_lenient(line), ensure_ascii=False) + "\n")
            rows += 1
            if limit is not None and rows >= limit:
                break
    if rows == 0:
        raise ValueError(f"No samples were materialized from {data_path}")
    return derived_path


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run reference-free MBR decoding for Gemma4 audio translation and evaluate BLEU/chrF/entity_lcs."
    )
    parser.add_argument("--test-data-path", type=Path, default=root_dir / "data" / "converted_testt_format" / "testt.jsonl")
    parser.add_argument("--test-entity-path", type=Path, default=root_dir / "data" / "converted_testt_format" / "testt.ner.jsonl")
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--processor-path", type=str, default=None)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-prefix-from", type=str, default="")
    parser.add_argument("--audio-prefix-to", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of audio samples per model.generate call.")
    parser.add_argument("--num-candidates", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    parser.add_argument("--mbr-bleu-weight", type=float, default=0.5)
    parser.add_argument("--mbr-chrf-weight", type=float, default=0.5)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--device-map", type=str, default="none")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument(
        "--prompt-mode",
        choices=["entity_focused", "dataset", "fixed"],
        default="entity_focused",
    )
    parser.add_argument("--prompt-text", type=str, default=ENTITY_FOCUSED_TRANSLATION_PROMPT)
    parser.add_argument("--entity-lcs-labels", type=str, default="LOC,PER,TERM,NUM,ORG,TIME")
    parser.add_argument(
        "--entity-lcs-skip-labels",
        type=str,
        default="",
        help=(
            "Comma-separated labels skipped in final entity recall. Empty means include all labels. "
            "This also applies when --entity-metric hard_key_recall."
        ),
    )
    parser.add_argument(
        "--entity-metric",
        choices=["lcs", "hard_key_recall"],
        default="hard_key_recall",
        help="Entity metric. hard_key_recall requires the full normalized reference entity to appear in prediction.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-existing", action="store_true")
    parser.add_argument("--config-out", type=Path, default=None)
    return parser.parse_args()


def _progress(iterable: Any, **kwargs: Any) -> Any:
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, **kwargs)


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _load_processor(processor_path: str) -> Any:
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(processor_path)
    except ValueError:
        try:
            return AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        except Exception:
            return _load_processor_by_declared_class(processor_path)


def _load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForImageTextToText

    processor_path = args.processor_path or args.base_model_path
    processor = _load_processor(processor_path)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": _resolve_torch_dtype(args.torch_dtype),
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    model = AutoModelForImageTextToText.from_pretrained(args.base_model_path, **model_kwargs)

    if args.adapter_path:
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("peft is required to load Gemma LoRA adapters.") from exc
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)

    if args.device_map.lower() == "none":
        model.to(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model, processor


def _sample_prompt(sample: Any, args: argparse.Namespace) -> str:
    if args.prompt_mode == "dataset":
        return str(sample.dataset_prompt or "").strip()
    return str(args.prompt_text or "").strip()


def _load_audio_array(path: str | None, sampling_rate: int) -> Any:
    if not path:
        raise ValueError("Sample has no audio path.")
    from eval_suite.audio import load_audio_array

    return load_audio_array(path, sampling_rate)


def generate_candidate_batch(model: Any, processor: Any, samples: list[Any], args: argparse.Namespace) -> list[list[str]]:
    conversations = []
    for sample in samples:
        content = [
            {
                "type": "audio",
                "audio": _load_audio_array(sample.audio_path, args.sampling_rate),
            }
        ]
        prompt = _sample_prompt(sample, args)
        if prompt:
            content.append({"type": "text", "text": prompt})
        conversations.append([{"role": "user", "content": content}])

    model_inputs = processor.apply_chat_template(
        conversations,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        processor_kwargs={"sampling_rate": args.sampling_rate},
    )
    if args.device_map.lower() == "none":
        model_inputs = _move_to_device(model_inputs, args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_return_sequences": args.num_candidates,
        "pad_token_id": _get_pad_token_id(processor),
    }
    if args.top_k is not None and args.top_k > 0:
        generation_kwargs["top_k"] = args.top_k
    if args.repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.no_repeat_ngram_size is not None:
        generation_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    with torch.inference_mode():
        generated = model.generate(**model_inputs, **generation_kwargs)

    prompt_length = model_inputs["input_ids"].shape[1]
    decoded = processor.batch_decode(
        generated[:, prompt_length:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    clean = [str(text).strip().replace("\n", " ") for text in decoded]
    grouped = []
    for index in range(len(samples)):
        start = index * args.num_candidates
        grouped.append(clean[start : start + args.num_candidates])
    return grouped


def pairwise_utility(left: str, right: str, bleu_weight: float, chrf_weight: float) -> float:
    if not left or not right:
        return 0.0
    sacrebleu = get_sacrebleu()
    bleu = sacrebleu.sentence_bleu(left, [right], tokenize="zh", use_effective_order=True).score / 100.0
    chrf = sacrebleu.sentence_chrf(left, [right], word_order=0).score / 100.0
    return bleu_weight * float(bleu) + chrf_weight * float(chrf)


def select_mbr_candidate(candidates: list[str], bleu_weight: float, chrf_weight: float) -> tuple[int, list[float]]:
    if not candidates:
        return 0, []
    if len(candidates) == 1:
        return 0, [0.0]
    scores: list[float] = []
    for i, candidate in enumerate(candidates):
        total = 0.0
        comparisons = 0
        for j, other in enumerate(candidates):
            if i == j:
                continue
            total += pairwise_utility(candidate, other, bleu_weight, chrf_weight)
            comparisons += 1
        scores.append(total / comparisons if comparisons else 0.0)
    return max(range(len(scores)), key=lambda index: scores[index]), scores


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = record.get("id")
            if sample_id:
                completed.add(str(sample_id))
    return completed


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def build_offline_eval_config(args: argparse.Namespace, prediction_path: Path) -> dict[str, Any]:
    entity_lcs_labels = [label.strip().upper() for label in args.entity_lcs_labels.split(",") if label.strip()]
    entity_lcs_skip_labels = [label.strip().upper() for label in args.entity_lcs_skip_labels.split(",") if label.strip()]
    entity_metric = "entity_key_recall" if args.entity_metric == "hard_key_recall" else "entity_lcs"
    dataset_path = Path(getattr(args, "eval_test_data_path", args.test_data_path))
    return {
        "dataset": {
            "path": str(dataset_path),
            "format": "converted_translation_jsonl",
            "reference_entity_path": str(args.test_entity_path),
            "audio_prefix_from": args.audio_prefix_from,
            "audio_prefix_to": args.audio_prefix_to,
            "limit": args.limit,
        },
        "model": {
            "kind": "offline_predictions",
            "name": args.model_name or f"{Path(args.base_model_path.rstrip('/')).name}-mbr",
            "batch_size": args.batch_size,
        },
        "evaluation": {
            "mode": "offline_predictions",
            "output_dir": str(args.output_dir),
            "prediction_path": str(prediction_path),
            "prediction_id_field": "id",
            "prediction_text_field": "prediction_text",
            "metrics": ["bleu", "chrf", entity_metric],
            "entity_lcs_labels": entity_lcs_labels,
            "entity_lcs_skip_labels": entity_lcs_skip_labels,
            "progress_every": args.progress_every,
        },
    }


def main() -> None:
    args = parse_args()
    if args.num_candidates < 1:
        raise ValueError("--num-candidates must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    weight_sum = args.mbr_bleu_weight + args.mbr_chrf_weight
    if weight_sum <= 0:
        raise ValueError("--mbr-bleu-weight + --mbr-chrf-weight must be > 0.")
    args.mbr_bleu_weight /= weight_sum
    args.mbr_chrf_weight /= weight_sum

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates_path = args.output_dir / "candidates.mbr.jsonl"
    predictions_path = args.output_dir / "predictions.mbr.jsonl"
    summary_path = args.output_dir / "metrics.mbr.json"
    config_path = args.config_out or (args.output_dir / "gemma4_mbr_eval.resolved.yaml")

    if not args.resume and not args.overwrite_existing:
        existing = [path for path in (candidates_path, predictions_path, summary_path) if path.exists()]
        if existing:
            raise FileExistsError(
                "Output files already exist. Use --resume to continue or --overwrite-existing to replace them: "
                + ", ".join(str(path) for path in existing)
            )
    if args.overwrite_existing and not args.resume:
        for path in (candidates_path, predictions_path, summary_path, config_path):
            if path.exists():
                path.unlink()

    sidecar_report = validate_testt_sidecar(args.test_data_path, args.test_entity_path, args.limit)
    (args.output_dir / "sidecar.validation.json").write_text(
        json.dumps(sidecar_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.eval_test_data_path = materialize_lenient_test_jsonl(args.test_data_path, args.output_dir, args.limit)

    eval_config = build_offline_eval_config(args, predictions_path)
    config_path.write_text(yaml.safe_dump(eval_config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    samples = load_samples(load_eval_spec(config_path).dataset)

    completed_ids = load_completed_ids(predictions_path) if args.resume else set()
    pending_samples = [sample for sample in samples if sample.sample_id not in completed_ids]

    if pending_samples:
        model, processor = _load_model(args)
        batches = _batched(pending_samples, args.batch_size)
        iterator = _progress(batches, total=len(batches), desc="MBR", unit="batch")
        done = len(completed_ids)
        for batch_index, batch in enumerate(iterator, start=1):
            grouped_candidates = generate_candidate_batch(model, processor, batch, args)
            for sample, candidates in zip(batch, grouped_candidates):
                selected_index, mbr_scores = select_mbr_candidate(
                    candidates,
                    args.mbr_bleu_weight,
                    args.mbr_chrf_weight,
                )
                selected_text = candidates[selected_index] if candidates else ""
                candidate_rows = [
                    {"index": index, "text": text, "mbr_score": mbr_scores[index] if index < len(mbr_scores) else 0.0}
                    for index, text in enumerate(candidates)
                ]
                write_jsonl_row(
                    candidates_path,
                    {
                        "id": sample.sample_id,
                        "audio_path": sample.audio_path,
                        "reference_text": sample.reference_text,
                        "selected_index": selected_index,
                        "selected_text": selected_text,
                        "mbr_bleu_weight": args.mbr_bleu_weight,
                        "mbr_chrf_weight": args.mbr_chrf_weight,
                        "candidates": candidate_rows,
                    },
                )
                write_jsonl_row(
                    predictions_path,
                    {
                        "id": sample.sample_id,
                        "prediction_text": selected_text,
                        "reference_text": sample.reference_text,
                        "audio_path": sample.audio_path,
                        "model_name": eval_config["model"]["name"],
                        "adapter_path": args.adapter_path,
                        "mbr_selected_index": selected_index,
                        "mbr_selected_score": mbr_scores[selected_index] if mbr_scores else 0.0,
                        "num_candidates": len(candidates),
                    },
                )
                done += 1
            if hasattr(iterator, "set_postfix"):
                iterator.set_postfix(samples=done)
            if args.progress_every > 0 and batch_index % args.progress_every == 0:
                print(f"MBR generated {done}/{len(samples)} samples", flush=True)

    summary = run_eval_spec(config_path)
    summary["entity_recall"] = summary.get(
        "entity_key_recall",
        summary.get("entity_recall", summary.get("entity_lcs_recall", 0.0)),
    )
    summary["entity_metric"] = args.entity_metric
    summary["metrics_requested"] = ["BLEU", "chrF", "Entity-Recall"]
    summary["mbr_num_candidates"] = args.num_candidates
    summary["mbr_bleu_weight"] = args.mbr_bleu_weight
    summary["mbr_chrf_weight"] = args.mbr_chrf_weight
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import random
import warnings
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from fca_grpo_group_risk_impl import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_PROMPT,
    PLACEHOLDER_TRANSLATION,
    attach_sidecar_keys,
    build_prompt_messages,
    compute_key_recall,
    compute_len_ratio,
    compute_reference_ce_loss,
    ensure_batch_dims,
    extract_audio_path_from_messages,
    extract_key_texts_from_record,
    extract_prompt_from_messages,
    extract_reference,
    extract_text_from_content,
    get_sacrebleu,
    get_vocab_size,
    load_key_sidecar,
    normalize_text,
    resolve_torch_dtype,
    rewrite_audio_path,
)

warnings.filterwarnings(
    "ignore",
    message=r"Kwargs passed to `processor\.__call__` have to be in `processor_kwargs` dict, not in `\*\*kwargs`",
)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate BLEU/chrF/key recall/len ratio/loss for Gemma4 GRPO experiments."
    )
    parser.add_argument("--base-model-path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=root_dir / "data" / "converted_testt_format" / "testt.jsonl",
    )
    parser.add_argument("--entity-path", type=Path, default=None)
    parser.add_argument(
        "--key-match-mode",
        type=str,
        default="normalized_exact",
        choices=("exact", "normalized_exact"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--objective", type=str, default=None)
    parser.add_argument("--bleu-weight", type=float, default=None)
    parser.add_argument("--chrf-weight", type=float, default=None)
    parser.add_argument("--key-weight", type=float, default=None)
    parser.add_argument("--ce-weight", type=float, default=None)
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    parser.add_argument("--audio-prefix-from", type=str, default="")
    parser.add_argument("--audio-prefix-to", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--device-map", type=str, default="none")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16"],
        default="no",
        help="Runtime dtype for reference CE scoring; defaults to no extra casting.",
    )
    parser.add_argument(
        "--use-deepspeed-zero3",
        action="store_true",
        help="Compatibility flag for CE scoring with the group-risk runtime.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--load-in-8bit", action="store_true")
    return parser.parse_args()


def set_random_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_record(
    record: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, Any] | None, int]:
    if {"key", "audio", "gt"}.issubset(record):
        reference = extract_reference(record["gt"])
        if reference == PLACEHOLDER_TRANSLATION:
            return None, 1
        return (
            {
                "id": record["key"],
                "audio_path": rewrite_audio_path(
                    record["audio"], args.audio_prefix_from, args.audio_prefix_to
                ),
                "prompt": args.prompt,
                "reference": normalize_text(reference),
                "gold_keys": extract_key_texts_from_record(record),
            },
            0,
        )

    if {"id", "messages"}.issubset(record):
        messages = record["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("Messages format requires at least user and assistant turns.")
        reference = extract_text_from_content(messages[-1].get("content"))
        if reference == PLACEHOLDER_TRANSLATION:
            return None, 1
        return (
            {
                "id": record["id"],
                "audio_path": rewrite_audio_path(
                    extract_audio_path_from_messages(messages),
                    args.audio_prefix_from,
                    args.audio_prefix_to,
                ),
                "prompt": extract_prompt_from_messages(messages),
                "reference": normalize_text(reference),
                "gold_keys": extract_key_texts_from_record(record),
            },
            0,
        )

    raise ValueError(
        "Unsupported input record format. Expected either key/audio/gt or id/messages."
    )


def load_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    samples: list[dict[str, Any]] = []
    skipped_placeholder = 0
    with args.data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample, skipped = parse_record(json.loads(line), args)
            skipped_placeholder += skipped
            if sample is None:
                continue
            samples.append(sample)
            if args.limit is not None and len(samples) >= args.limit:
                break
    if not samples:
        raise ValueError(f"No valid evaluation samples were loaded from {args.data_path}.")
    return samples, skipped_placeholder


def load_experiment_metadata(adapter_path: Path | None) -> dict[str, Any]:
    if adapter_path is None:
        return {}
    for candidate in (adapter_path.parent / "run_config.json", adapter_path / "run_config.json"):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {}


def infer_experiment_name(
    explicit_name: str | None,
    adapter_path: Path | None,
    metadata: dict[str, Any],
) -> str:
    if explicit_name:
        return explicit_name
    if metadata.get("experiment_name"):
        return str(metadata["experiment_name"])
    if adapter_path is not None:
        return adapter_path.parent.name if adapter_path.name.startswith("adapter_") else adapter_path.name
    return "base_model_eval"


def load_model_and_processor(args: argparse.Namespace):
    processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
        "low_cpu_mem_usage": True,
    }
    if args.device_map != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForImageTextToText.from_pretrained(args.base_model_path, trust_remote_code=True, **model_kwargs)

    if args.adapter_path is not None:
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("peft is required to evaluate a LoRA adapter.") from exc
        model = PeftModel.from_pretrained(model, str(args.adapter_path), is_trainable=False)
        extra_projector_path = args.adapter_path / "extra_trainable_projector.pt"
        if extra_projector_path.exists():
            extra_state = torch.load(extra_projector_path, map_location="cpu")
            named_parameters = dict(model.named_parameters())
            missing: list[str] = []
            with torch.no_grad():
                for name, tensor in extra_state.items():
                    parameter = named_parameters.get(name)
                    if parameter is None:
                        missing.append(name)
                        continue
                    parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            if missing:
                print(
                    json.dumps(
                        {
                            "event": "missing_extra_projector_parameters",
                            "count": len(missing),
                            "sample": missing[:10],
                        },
                        ensure_ascii=False,
                    )
                )

    model.eval()
    if args.device_map == "none":
        target_device = args.device or "cuda:0"
        if target_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"Requested device {target_device}, but CUDA is not available."
            )
        model.to(target_device)
    return processor, model


def infer_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def get_pad_token_id(processor) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return tokenizer.eos_token_id


def get_generation_stop_token_ids(processor) -> int | list[int] | None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None
    stop_ids: list[int] = []
    if tokenizer.eos_token_id is not None:
        stop_ids.append(int(tokenizer.eos_token_id))
    for token in ("<turn|>", "<end_of_turn>", "<|end_of_turn|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_id:
            stop_ids.append(token_id)
    stop_ids = list(dict.fromkeys(stop_ids))
    if not stop_ids:
        return None
    return stop_ids[0] if len(stop_ids) == 1 else stop_ids


def generate_predictions(
    samples: list[dict[str, Any]],
    processor,
    model,
    args: argparse.Namespace,
) -> list[str]:
    predictions: list[str] = []
    device = infer_device(model)
    pad_token_id = get_pad_token_id(processor)
    eos_token_id = get_generation_stop_token_ids(processor)
    batch_starts = list(range(0, len(samples), args.batch_size))
    progress_bar = tqdm(
        batch_starts,
        desc="eval-generate",
        leave=False,
    )
    for start in progress_bar:
        batch = samples[start : start + args.batch_size]
        prompt_messages = [build_prompt_messages(sample, args.sampling_rate) for sample in batch]
        model_inputs = processor.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            processor_kwargs={"sampling_rate": args.sampling_rate},
        )
        model_inputs = ensure_batch_dims(model_inputs)
        model_inputs = move_batch_to_device(model_inputs, device)
        with torch.inference_mode():
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "pad_token_id": pad_token_id,
            }
            if eos_token_id is not None:
                generation_kwargs["eos_token_id"] = eos_token_id
            if args.do_sample:
                generation_kwargs["temperature"] = args.temperature
                generation_kwargs["top_p"] = args.top_p
            generated = model.generate(
                **model_inputs,
                **generation_kwargs,
            )
        prompt_length = model_inputs["input_ids"].shape[1]
        generated_only = generated[:, prompt_length:]
        batch_predictions = processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        predictions.extend(normalize_text(text) for text in batch_predictions)
    progress_bar.close()
    return predictions


def save_predictions(predictions_path: Path, samples: list[dict[str, Any]], predictions: list[str]) -> None:
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8", newline="\n") as f:
        for sample, prediction in zip(samples, predictions):
            f.write(
                json.dumps(
                    {
                        "id": sample["id"],
                        "audio_path": sample["audio_path"],
                        "prompt": sample["prompt"],
                        "reference": sample["reference"],
                        "prediction": prediction,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def main() -> None:
    args = parse_args()
    set_random_seed(args.seed)
    metadata = load_experiment_metadata(args.adapter_path)
    experiment_name = infer_experiment_name(args.experiment_name, args.adapter_path, metadata)
    output_dir = args.output_dir or (Path(__file__).resolve().parents[1] / "eval_outputs" / experiment_name)
    predictions_out = output_dir / "testt_predictions.jsonl"
    metrics_out = output_dir / "testt_metrics.json"

    samples, skipped_placeholder = load_samples(args)
    samples = attach_sidecar_keys(samples, load_key_sidecar(args.entity_path))
    key_labels_available = args.entity_path is not None
    processor, model = load_model_and_processor(args)
    predictions = generate_predictions(samples, processor, model, args)

    sacrebleu = get_sacrebleu()
    references = [sample["reference"] for sample in samples]
    bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="zh").score
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=0).score
    eval_key_recall = (
        sum(
            compute_key_recall(
                prediction,
                sample.get("gold_keys", []),
                match_mode=args.key_match_mode,
            )
            for prediction, sample in zip(predictions, samples)
        )
        / len(samples)
        if key_labels_available
        else 0.0
    )
    eval_len_ratio = sum(
        compute_len_ratio(prediction, reference)
        for prediction, reference in zip(predictions, references)
    ) / len(samples)
    vocab_size = get_vocab_size(processor)
    device = infer_device(model)
    eval_loss_values: list[float] = []
    for sample in samples:
        _, normalized_ce = compute_reference_ce_loss(
            model=model,
            processor=processor,
            sample=sample,
            accelerator=type("EvalAccelerator", (), {"device": device})(),
            args=args,
            vocab_size=vocab_size,
        )
        eval_loss_values.append(float(normalized_ce.detach().cpu().item()))
    eval_loss = sum(eval_loss_values) / len(eval_loss_values)

    weights = metadata.get("normalized_weights", {})
    metrics = {
        "experiment_name": experiment_name,
        "base_model_path": args.base_model_path,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "checkpoint_path": str(args.adapter_path) if args.adapter_path else args.base_model_path,
        "data_path": str(args.data_path),
        "entity_path": str(args.entity_path) if args.entity_path else None,
        "num_samples": len(samples),
        "skipped_placeholder": skipped_placeholder,
        "objective": args.objective or metadata.get("objective"),
        "bleu_weight": args.bleu_weight if args.bleu_weight is not None else weights.get("bleu"),
        "chrf_weight": args.chrf_weight if args.chrf_weight is not None else weights.get("chrf"),
        "key_weight": args.key_weight if args.key_weight is not None else weights.get("key"),
        "ce_weight": args.ce_weight if args.ce_weight is not None else weights.get("ce"),
        "seed": args.seed,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "eval_bleu": bleu,
        "eval_chrf": chrf,
        "eval_key_recall": eval_key_recall,
        "eval_len_ratio": eval_len_ratio,
        "eval_loss": eval_loss,
        "predictions_out": str(predictions_out),
        "key_labels_available": key_labels_available,
        "key_match_mode": args.key_match_mode,
    }
    save_predictions(predictions_out, samples, predictions)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

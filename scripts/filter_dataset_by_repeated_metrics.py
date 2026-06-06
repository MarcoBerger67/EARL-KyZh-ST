from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from eval_testt_bleu_chrf import load_model_and_processor, parse_record
from fca_grpo_group_risk_impl import compute_sentence_scores, normalize_text


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Filter a JSONL dataset using repeated sentence BLEU/chrF metrics from an SFT model."
    )
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=(
            root_dir
            / "data"
            / "converted_testt_format"
            / "train_ky2zh_full285h_stage3.cleaned.final.jsonl"
        ),
    )
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--removed-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--csv-path", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-max-bleu", type=float, default=10.0)
    parser.add_argument("--min-max-chrf", type=float, default=18.0)
    parser.add_argument("--min-std-bleu", type=float, default=3.0)
    parser.add_argument("--min-std-chrf", type=float, default=2.0)
    parser.add_argument("--max-std-bleu", type=float, default=None)
    parser.add_argument("--max-std-chrf", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--audio-prefix-from", type=str, default="")
    parser.add_argument("--audio-prefix-to", type=str, default="")
    parser.add_argument("--do-sample", action="store_true")
    parser.set_defaults(do_sample=True)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_audio_and_prompt(sample: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": sample["audio_path"]},
                {"type": "text", "text": sample["prompt"]},
            ],
        }
    ]


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


def infer_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def generate_once(
    sample: dict[str, Any],
    processor,
    model,
    args: argparse.Namespace,
) -> str:
    device = infer_device(model)
    pad_token_id = get_pad_token_id(processor)
    prompt_messages = [extract_audio_and_prompt(sample)]
    model_inputs = processor.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        processor_kwargs={"sampling_rate": args.sampling_rate},
    )
    model_inputs = move_batch_to_device(model_inputs, device)
    with torch.inference_mode():
        generated = model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=pad_token_id,
        )
    prompt_length = model_inputs["input_ids"].shape[1]
    generated_only = generated[:, prompt_length:]
    text = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return text.strip()


def main() -> None:
    args = parse_args()
    if args.runs < 3:
        raise ValueError("--runs must be at least 3.")

    raw_rows: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    skipped_placeholder = 0
    with args.data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw_record = json.loads(line)
            sample, skipped = parse_record(raw_record, args)
            skipped_placeholder += skipped
            if sample is None:
                continue
            raw_rows.append(raw_record)
            samples.append(sample)
            if args.limit is not None and len(samples) >= args.limit:
                break
    if not samples:
        raise ValueError(f"No valid filtering samples were loaded from {args.data_path}.")

    processor, model = load_model_and_processor(args)
    kept_rows: list[dict[str, Any]] = []
    removed_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    progress = tqdm(samples, desc="filter-dataset", leave=False)
    for sample_idx, sample in enumerate(progress):
        raw_record = raw_rows[sample_idx]
        bleu_scores: list[float] = []
        chrf_scores: list[float] = []
        predictions: list[str] = []
        for run_idx in range(args.runs):
            set_random_seed(args.seed + sample_idx * args.runs + run_idx)
            prediction = normalize_text(generate_once(sample, processor, model, args))
            bleu, chrf = compute_sentence_scores(prediction, sample["reference"])
            predictions.append(prediction)
            bleu_scores.append(bleu * 100.0)
            chrf_scores.append(chrf * 100.0)

        mean_bleu = statistics.fmean(bleu_scores)
        mean_chrf = statistics.fmean(chrf_scores)
        max_bleu = max(bleu_scores)
        max_chrf = max(chrf_scores)
        std_bleu = statistics.pstdev(bleu_scores)
        std_chrf = statistics.pstdev(chrf_scores)
        keep = (
            max_bleu >= args.min_max_bleu
            and max_chrf >= args.min_max_chrf
            and std_bleu >= args.min_std_bleu
            and std_chrf >= args.min_std_chrf
            and (args.max_std_bleu is None or std_bleu <= args.max_std_bleu)
            and (args.max_std_chrf is None or std_chrf <= args.max_std_chrf)
        )
        row = {
            "id": sample["id"],
            "audio_path": sample["audio_path"],
            "reference": sample["reference"],
            "mean_bleu": mean_bleu,
            "max_bleu": max_bleu,
            "std_bleu": std_bleu,
            "mean_chrf": mean_chrf,
            "max_chrf": max_chrf,
            "std_chrf": std_chrf,
            "keep": keep,
            "predictions": predictions,
        }
        metric_rows.append(row)
        if keep:
            kept_rows.append(raw_record)
        else:
            removed_rows.append(raw_record)

        if args.progress_every > 0 and (sample_idx + 1) % args.progress_every == 0:
            progress.set_postfix(
                kept=len(kept_rows),
                removed=len(removed_rows),
            )
    progress.close()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    for path, rows in ((args.output_path, kept_rows), (args.removed_path, removed_rows)):
        with path.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "audio_path",
                "mean_bleu",
                "max_bleu",
                "std_bleu",
                "mean_chrf",
                "max_chrf",
                "std_chrf",
                "keep",
            ],
        )
        writer.writeheader()
        for row in metric_rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    report = {
        "base_model_path": args.base_model_path,
        "adapter_path": str(args.adapter_path),
        "source_data_path": str(args.data_path),
        "output_path": str(args.output_path),
        "removed_path": str(args.removed_path),
        "csv_path": str(args.csv_path),
        "num_samples": len(samples),
        "skipped_placeholder": skipped_placeholder,
        "runs": args.runs,
        "seed": args.seed,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "min_max_bleu": args.min_max_bleu,
        "min_max_chrf": args.min_max_chrf,
        "min_std_bleu": args.min_std_bleu,
        "min_std_chrf": args.min_std_chrf,
        "max_std_bleu": args.max_std_bleu,
        "max_std_chrf": args.max_std_chrf,
        "kept_rows": len(kept_rows),
        "removed_rows": len(removed_rows),
        "kept_ratio": len(kept_rows) / len(samples) if samples else 0.0,
        "mean_of_mean_bleu": statistics.fmean(row["mean_bleu"] for row in metric_rows) if metric_rows else 0.0,
        "mean_of_mean_chrf": statistics.fmean(row["mean_chrf"] for row in metric_rows) if metric_rows else 0.0,
        "mean_of_max_bleu": statistics.fmean(row["max_bleu"] for row in metric_rows) if metric_rows else 0.0,
        "mean_of_max_chrf": statistics.fmean(row["max_chrf"] for row in metric_rows) if metric_rows else 0.0,
        "mean_of_std_bleu": statistics.fmean(row["std_bleu"] for row in metric_rows) if metric_rows else 0.0,
        "mean_of_std_chrf": statistics.fmean(row["std_chrf"] for row in metric_rows) if metric_rows else 0.0,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

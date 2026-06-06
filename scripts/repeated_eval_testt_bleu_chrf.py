from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any

import torch

from eval_testt_bleu_chrf import (
    generate_predictions,
    infer_experiment_name,
    load_experiment_metadata,
    load_model_and_processor,
    load_samples,
    save_predictions,
)
from fca_grpo_group_risk_impl import (
    attach_sidecar_keys,
    compute_key_recall,
    compute_len_ratio,
    load_key_sidecar,
)
from train_gemma4_sft_qlora import get_sacrebleu


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run BLEU/chrF evaluation multiple times and summarize mean/std."
    )
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=root_dir / "data" / "converted_testt_format" / "testt.jsonl",
    )
    parser.add_argument("--entity-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--key-match-mode",
        type=str,
        default="normalized_exact",
        choices=("exact", "normalized_exact"),
    )
    parser.add_argument("--audio-prefix-from", type=str, default="")
    parser.add_argument("--audio-prefix-to", type=str, default="")
    parser.add_argument("--min-mean-bleu", type=float, default=0.0)
    parser.add_argument("--min-mean-chrf", type=float, default=0.0)
    parser.add_argument("--max-std-bleu", type=float, default=100.0)
    parser.add_argument("--max-std-chrf", type=float, default=100.0)
    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if args.runs < 3:
        raise ValueError("--runs must be at least 3.")

    metadata = load_experiment_metadata(args.adapter_path)
    experiment_name = infer_experiment_name(args.experiment_name, args.adapter_path, metadata)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples, skipped_placeholder = load_samples(args)
    samples = attach_sidecar_keys(samples, load_key_sidecar(args.entity_path))
    key_labels_available = args.entity_path is not None

    processor, model = load_model_and_processor(args)
    sacrebleu = get_sacrebleu()
    references = [sample["reference"] for sample in samples]
    run_rows: list[dict[str, Any]] = []

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        set_random_seed(run_seed)
        predictions = generate_predictions(samples, processor, model, args)
        predictions_out = output_dir / f"testt_predictions.run{run_idx + 1}.jsonl"
        save_predictions(predictions_out, samples, predictions)

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
        run_rows.append(
            {
                "run_index": run_idx + 1,
                "seed": run_seed,
                "eval_bleu": bleu,
                "eval_chrf": chrf,
                "eval_key_recall": eval_key_recall,
                "eval_len_ratio": eval_len_ratio,
                "predictions_out": str(predictions_out),
            }
        )

    bleu_scores = [row["eval_bleu"] for row in run_rows]
    chrf_scores = [row["eval_chrf"] for row in run_rows]
    bleu_mean = statistics.fmean(bleu_scores)
    chrf_mean = statistics.fmean(chrf_scores)
    bleu_std = statistics.pstdev(bleu_scores)
    chrf_std = statistics.pstdev(chrf_scores)
    passed = (
        bleu_mean >= args.min_mean_bleu
        and chrf_mean >= args.min_mean_chrf
        and bleu_std <= args.max_std_bleu
        and chrf_std <= args.max_std_chrf
    )

    summary = {
        "experiment_name": experiment_name,
        "base_model_path": args.base_model_path,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "data_path": str(args.data_path),
        "entity_path": str(args.entity_path) if args.entity_path else None,
        "num_samples": len(samples),
        "skipped_placeholder": skipped_placeholder,
        "runs": args.runs,
        "base_seed": args.seed,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "mean_bleu": bleu_mean,
        "std_bleu": bleu_std,
        "mean_chrf": chrf_mean,
        "std_chrf": chrf_std,
        "min_mean_bleu": args.min_mean_bleu,
        "min_mean_chrf": args.min_mean_chrf,
        "max_std_bleu": args.max_std_bleu,
        "max_std_chrf": args.max_std_chrf,
        "passed_thresholds": passed,
        "run_metrics": run_rows,
    }

    (output_dir / "repeated_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

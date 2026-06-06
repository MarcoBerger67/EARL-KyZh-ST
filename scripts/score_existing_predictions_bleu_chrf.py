from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path
from typing import Any

from fca_grpo_group_risk_impl import compute_sentence_scores
from train_gemma4_sft_qlora import get_sacrebleu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score existing prediction JSONL files without running generation."
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        action="append",
        default=[],
        help="Prediction JSONL path. Can be passed multiple times.",
    )
    parser.add_argument(
        "--predictions-glob",
        type=str,
        default=None,
        help="Optional glob pattern for prediction JSONL files, e.g. 'eval_outputs/exp/testt_predictions.run*.jsonl'.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional summary JSON output path.",
    )
    parser.add_argument(
        "--sample-stats-csv",
        type=Path,
        default=None,
        help="Optional per-sample CSV with sentence-level mean/max/std stats across runs.",
    )
    parser.add_argument(
        "--sample-stats-jsonl",
        type=Path,
        default=None,
        help="Optional per-sample JSONL with sentence-level mean/max/std stats across runs.",
    )
    parser.add_argument(
        "--source-data-path",
        type=Path,
        default=None,
        help="Optional source dataset JSONL used to materialize kept/removed records by id.",
    )
    parser.add_argument(
        "--filtered-output-path",
        type=Path,
        default=None,
        help="Optional kept dataset JSONL output path.",
    )
    parser.add_argument(
        "--filtered-removed-path",
        type=Path,
        default=None,
        help="Optional removed dataset JSONL output path.",
    )
    parser.add_argument("--min-max-bleu", type=float, default=None)
    parser.add_argument("--min-max-chrf", type=float, default=None)
    parser.add_argument("--min-std-bleu", type=float, default=None)
    parser.add_argument("--min-std-chrf", type=float, default=None)
    parser.add_argument("--max-std-bleu", type=float, default=None)
    parser.add_argument("--max-std-chrf", type=float, default=None)
    return parser.parse_args()


def resolve_prediction_paths(args: argparse.Namespace) -> list[Path]:
    resolved: list[Path] = list(args.predictions_path)
    if args.predictions_glob:
        resolved.extend(Path(path) for path in sorted(glob.glob(args.predictions_glob)))
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in resolved:
        normalized = path.resolve()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_paths.append(normalized)
    if not unique_paths:
        raise ValueError("Provide at least one --predictions-path or --predictions-glob.")
    for path in unique_paths:
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")
    return unique_paths


def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for required_key in ("id", "reference", "prediction"):
                if required_key not in row:
                    raise ValueError(
                        f"{path}:{line_no} is missing required key '{required_key}'."
                    )
            rows.append(row)
    if not rows:
        raise ValueError(f"No valid prediction rows found in {path}.")
    return rows


def validate_aligned_runs(run_rows: list[tuple[Path, list[dict[str, Any]]]]) -> None:
    base_path, base_rows = run_rows[0]
    base_pairs = [(row["id"], row["reference"]) for row in base_rows]
    for path, rows in run_rows[1:]:
        pairs = [(row["id"], row["reference"]) for row in rows]
        if pairs != base_pairs:
            raise ValueError(
                f"Prediction files are not aligned between {base_path} and {path}."
            )


def compute_corpus_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    sacrebleu = get_sacrebleu()
    predictions = [str(row["prediction"]) for row in rows]
    references = [str(row["reference"]) for row in rows]
    bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="zh").score
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=0).score
    return {"eval_bleu": float(bleu), "eval_chrf": float(chrf)}


def compute_sample_stats(run_rows: list[tuple[Path, list[dict[str, Any]]]]) -> list[dict[str, Any]]:
    sample_stats: list[dict[str, Any]] = []
    num_runs = len(run_rows)
    for row_index in range(len(run_rows[0][1])):
        first_row = run_rows[0][1][row_index]
        bleu_scores: list[float] = []
        chrf_scores: list[float] = []
        predictions: list[str] = []
        source_files: list[str] = []
        for path, rows in run_rows:
            row = rows[row_index]
            bleu, chrf = compute_sentence_scores(str(row["prediction"]), str(row["reference"]))
            bleu_scores.append(bleu * 100.0)
            chrf_scores.append(chrf * 100.0)
            predictions.append(str(row["prediction"]))
            source_files.append(str(path))
        sample_stats.append(
            {
                "id": first_row["id"],
                "reference": first_row["reference"],
                "runs": num_runs,
                "mean_bleu": statistics.fmean(bleu_scores),
                "max_bleu": max(bleu_scores),
                "std_bleu": statistics.pstdev(bleu_scores),
                "mean_chrf": statistics.fmean(chrf_scores),
                "max_chrf": max(chrf_scores),
                "std_chrf": statistics.pstdev(chrf_scores),
                "predictions": predictions,
                "source_files": source_files,
            }
        )
    return sample_stats


def write_sample_stats_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "mean_bleu",
                "max_bleu",
                "std_bleu",
                "mean_chrf",
                "max_chrf",
                "std_chrf",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def write_sample_stats_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_source_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = row.get("id") or row.get("key")
            if not record_id:
                raise ValueError(f"{path}:{line_no} is missing 'id'/'key'.")
            record_key = str(record_id)
            if record_key in records:
                raise ValueError(f"Duplicate record id in source dataset: {record_key}")
            records[record_key] = row
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def should_filter(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.filtered_output_path,
            args.filtered_removed_path,
            args.min_max_bleu,
            args.min_max_chrf,
            args.min_std_bleu,
            args.min_std_chrf,
            args.max_std_bleu,
            args.max_std_chrf,
        )
    )


def validate_filter_args(args: argparse.Namespace) -> None:
    if not should_filter(args):
        return
    required_missing: list[str] = []
    if args.source_data_path is None:
        required_missing.append("--source-data-path")
    if args.filtered_output_path is None:
        required_missing.append("--filtered-output-path")
    if args.filtered_removed_path is None:
        required_missing.append("--filtered-removed-path")
    if args.min_max_bleu is None:
        required_missing.append("--min-max-bleu")
    if args.min_max_chrf is None:
        required_missing.append("--min-max-chrf")
    if args.min_std_bleu is None:
        required_missing.append("--min-std-bleu")
    if args.min_std_chrf is None:
        required_missing.append("--min-std-chrf")
    if required_missing:
        raise ValueError(
            "Filtering requires all of these arguments: " + ", ".join(required_missing)
        )


def main() -> None:
    args = parse_args()
    validate_filter_args(args)
    prediction_paths = resolve_prediction_paths(args)
    run_rows = [(path, load_prediction_rows(path)) for path in prediction_paths]
    validate_aligned_runs(run_rows)

    per_run_metrics: list[dict[str, Any]] = []
    for path, rows in run_rows:
        metrics = compute_corpus_metrics(rows)
        per_run_metrics.append(
            {
                "predictions_path": str(path),
                "num_samples": len(rows),
                **metrics,
            }
        )

    bleu_scores = [row["eval_bleu"] for row in per_run_metrics]
    chrf_scores = [row["eval_chrf"] for row in per_run_metrics]
    summary: dict[str, Any] = {
        "num_runs": len(per_run_metrics),
        "per_run_metrics": per_run_metrics,
        "mean_bleu": statistics.fmean(bleu_scores),
        "max_bleu": max(bleu_scores),
        "std_bleu": statistics.pstdev(bleu_scores),
        "mean_chrf": statistics.fmean(chrf_scores),
        "max_chrf": max(chrf_scores),
        "std_chrf": statistics.pstdev(chrf_scores),
    }

    sample_stats: list[dict[str, Any]] | None = None
    if len(run_rows) > 1 and (
        args.sample_stats_csv or args.sample_stats_jsonl or should_filter(args)
    ):
        sample_stats = compute_sample_stats(run_rows)
        summary["sample_stats_rows"] = len(sample_stats)
        summary["mean_of_mean_bleu"] = statistics.fmean(row["mean_bleu"] for row in sample_stats)
        summary["mean_of_max_bleu"] = statistics.fmean(row["max_bleu"] for row in sample_stats)
        summary["mean_of_std_bleu"] = statistics.fmean(row["std_bleu"] for row in sample_stats)
        summary["mean_of_mean_chrf"] = statistics.fmean(row["mean_chrf"] for row in sample_stats)
        summary["mean_of_max_chrf"] = statistics.fmean(row["max_chrf"] for row in sample_stats)
        summary["mean_of_std_chrf"] = statistics.fmean(row["std_chrf"] for row in sample_stats)
        if args.sample_stats_csv:
            write_sample_stats_csv(args.sample_stats_csv, sample_stats)
        if args.sample_stats_jsonl:
            write_sample_stats_jsonl(args.sample_stats_jsonl, sample_stats)

    if should_filter(args):
        if len(run_rows) < 2:
            raise ValueError("Filtering on std/max requires at least two prediction runs.")
        assert sample_stats is not None
        source_records = load_source_records(args.source_data_path)
        kept_rows: list[dict[str, Any]] = []
        removed_rows: list[dict[str, Any]] = []
        kept_count = 0
        removed_count = 0
        for row in sample_stats:
            keep = (
                row["max_bleu"] >= args.min_max_bleu
                and row["max_chrf"] >= args.min_max_chrf
                and row["std_bleu"] >= args.min_std_bleu
                and row["std_chrf"] >= args.min_std_chrf
                and (args.max_std_bleu is None or row["std_bleu"] <= args.max_std_bleu)
                and (args.max_std_chrf is None or row["std_chrf"] <= args.max_std_chrf)
            )
            row["keep"] = keep
            source_row = source_records.get(str(row["id"]))
            if source_row is None:
                raise ValueError(f"Sample id {row['id']} not found in source dataset.")
            if keep:
                kept_rows.append(source_row)
                kept_count += 1
            else:
                removed_rows.append(source_row)
                removed_count += 1
        write_jsonl(args.filtered_output_path, kept_rows)
        write_jsonl(args.filtered_removed_path, removed_rows)
        if args.sample_stats_csv:
            write_sample_stats_csv(args.sample_stats_csv, sample_stats)
        if args.sample_stats_jsonl:
            write_sample_stats_jsonl(args.sample_stats_jsonl, sample_stats)
        summary["filtering"] = {
            "source_data_path": str(args.source_data_path),
            "filtered_output_path": str(args.filtered_output_path),
            "filtered_removed_path": str(args.filtered_removed_path),
            "min_max_bleu": args.min_max_bleu,
            "min_max_chrf": args.min_max_chrf,
            "min_std_bleu": args.min_std_bleu,
            "min_std_chrf": args.min_std_chrf,
            "max_std_bleu": args.max_std_bleu,
            "max_std_chrf": args.max_std_chrf,
            "kept_rows": kept_count,
            "removed_rows": removed_count,
            "kept_ratio": kept_count / len(sample_stats) if sample_stats else 0.0,
        }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

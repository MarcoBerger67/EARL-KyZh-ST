from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from eval_suite.runner import run_eval_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute BLEU/chrF/entity recall from an existing predictions.jsonl.")
    parser.add_argument("--test-data-path", type=Path, required=True)
    parser.add_argument("--test-entity-path", type=Path, required=True)
    parser.add_argument("--prediction-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", type=str, default="offline_predictions")
    parser.add_argument(
        "--entity-metric",
        choices=["lcs", "hard_key_recall"],
        default="lcs",
        help="Entity recall metric. hard_key_recall requires full normalized entity substring match.",
    )
    parser.add_argument("--entity-lcs-labels", type=str, default="LOC,PER,TERM,NUM,ORG,TIME")
    parser.add_argument(
        "--entity-lcs-skip-labels",
        type=str,
        default="",
        help='Comma-separated labels skipped in overall entity_lcs. Default "" includes all labels.',
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


def split_labels(value: str) -> list[str]:
    return [label.strip().upper() for label in value.split(",") if label.strip()]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    entity_metric_name = "entity_key_recall" if args.entity_metric == "hard_key_recall" else "entity_lcs"
    config: dict[str, Any] = {
        "dataset": {
            "path": str(args.test_data_path),
            "format": "converted_translation_jsonl",
            "reference_entity_path": str(args.test_entity_path),
            "limit": args.limit,
        },
        "model": {
            "kind": "offline_predictions",
            "name": args.model_name,
        },
        "evaluation": {
            "mode": "offline_predictions",
            "output_dir": str(args.output_dir),
            "prediction_path": str(args.prediction_path),
            "prediction_id_field": "id",
            "prediction_text_field": "prediction_text",
            "metrics": ["bleu", "chrf", entity_metric_name],
            "entity_lcs_labels": split_labels(args.entity_lcs_labels),
            "entity_lcs_skip_labels": split_labels(args.entity_lcs_skip_labels),
            "progress_every": args.progress_every,
        },
    }
    config_path = args.output_dir / "offline_metrics.resolved.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    summary = run_eval_spec(config_path)
    summary["entity_recall"] = summary.get("entity_recall", summary.get("entity_lcs_recall", 0.0))
    summary["metrics_requested"] = ["BLEU", "chrF", "Entity-Recall"]
    summary_path = args.output_dir / "metrics.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

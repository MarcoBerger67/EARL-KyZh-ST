from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_suite.runner import run_eval_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the independent multi-model BLEU/chrF/entity evaluation suite."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to evaluation YAML spec.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_eval_spec(args.config)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

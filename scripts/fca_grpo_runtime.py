from __future__ import annotations

"""Unified GRPO entrypoint.

This file keeps the public GRPO runtime name while exposing two selectable
objectives (choose one explicitly with --grpo-objective for reproduction):

- group_relative_risk_kl: implemented by the
  KL-regularized group-relative risk runtime.
- clipped_grpo: the PPO-style clipped-ratio objective, implemented in
  fca_grpo_clipped_runtime.
"""

import argparse
import sys
from collections.abc import Sequence


DEFAULT_GRPO_OBJECTIVE = "group_relative_risk_kl"
SUPPORTED_GRPO_OBJECTIVES = ("group_relative_risk_kl", "clipped_grpo")


def _extract_objective(argv: Sequence[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--grpo-objective",
        "--objective-variant",
        choices=SUPPORTED_GRPO_OBJECTIVES,
        default=DEFAULT_GRPO_OBJECTIVE,
    )
    namespace, remaining = parser.parse_known_args(list(argv))
    return namespace.grpo_objective, remaining


def train() -> None:
    objective, remaining = _extract_objective(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    if objective == "group_relative_risk_kl":
        from fca_grpo_risk_runtime import train as train_group_relative_risk

        train_group_relative_risk()
        return

    if objective == "clipped_grpo":
        from fca_grpo_clipped_runtime import train as train_clipped_grpo

        train_clipped_grpo()
        return

    raise ValueError(f"Unsupported GRPO objective: {objective}")


if __name__ == "__main__":
    train()

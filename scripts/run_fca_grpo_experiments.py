from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_ORDER = [
    "stkg_stage1_bleu_chrf",
    "stkg_stage2_full",
    "stkg_wo_stage1_stage2_from_sft",
    "stkg_stage1_bleu_only",
    "stkg_stage2_chrf_only",
    "stkg_stage2_chrf_entity_em",
    "stkg_stage2_tau04",
    "stkg_stage2_tau08",
    "grpo_bleu_only",
    "grpo_chrf_only",
    "grpo_bleu_ner_bleu_high",
]


def detect_repo_root() -> Path:
    candidates: list[Path] = []
    env_root = os.environ.get("GEMMA4_REPO_ROOT") or os.environ.get("GEMMA4_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path(__file__).absolute().parents[1])
    candidates.append(Path(__file__).resolve().parents[1])
    candidates.append(Path.cwd())

    for candidate in candidates:
        if (candidate / "scripts" / "run_fca_grpo_experiments.py").exists() and (
            candidate / "configs" / "fca_grpo" / "common.yaml"
        ).exists():
            return candidate
    return Path(__file__).absolute().parents[1]


def parse_args() -> argparse.Namespace:
    root_dir = detect_repo_root()
    parser = argparse.ArgumentParser(
        description="Run reproducible Gemma4 FCA-GRPO experiments from YAML configs."
    )
    parser.add_argument("command", choices=["run-sft", "dry-run-sft", "run-one", "run-all", "dry-run"])
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--sft-experiment-name", type=str, default=None)
    parser.add_argument("--sft-best-checkpoint", type=Path, default=None)
    parser.add_argument("--base-model-path", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--config-root", type=Path, default=root_dir / "configs" / "fca_grpo")
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Append this suffix to generated SFT/GRPO experiment directory names.",
    )
    parser.add_argument(
        "--auto-run-suffix",
        action="store_true",
        help="Append a timestamp suffix to generated experiment directory names.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Allow writing into an existing non-empty experiment directory.",
    )
    parser.add_argument("--train-data-path", type=Path, default=None)
    parser.add_argument("--train-entity-path", type=Path, default=None)
    parser.add_argument("--val-data-path", type=Path, default=None)
    parser.add_argument("--val-entity-path", type=Path, default=None)
    parser.add_argument("--test-data-path", type=Path, default=None)
    parser.add_argument("--test-entity-path", type=Path, default=None)
    parser.add_argument("--audio-prefix-from", type=str, default=None)
    parser.add_argument("--audio-prefix-to", type=str, default=None)
    parser.add_argument("--key-match-mode", type=str, default=None, choices=("exact", "normalized_exact", "fuzzy_substring"))
    parser.add_argument(
        "--entity-reward-mode",
        type=str,
        default=None,
        choices=("none", "key_recall", "entity_em", "entity_soft", "entity_gemma_fuzzy", "entity_substring"),
    )
    parser.add_argument("--entity-soft-tau", type=float, default=None)
    parser.add_argument("--entity-include-per", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--entity-embedding-model", type=str, default=None)
    parser.add_argument("--entity-embedding-max-length", type=int, default=None)
    parser.add_argument("--entity-embedding-pooling", type=str, default=None, choices=("mean", "cls"))
    parser.add_argument("--entity-embedding-device", type=str, default=None)
    parser.add_argument("--entity-extractor-device", type=str, default=None)
    parser.add_argument("--ner-tokenizer-model", type=str, default=None)
    parser.add_argument("--ner-model", type=str, default=None)
    parser.add_argument("--ner-tokenizer-path", type=str, default=None)
    parser.add_argument("--ner-path", type=str, default=None)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument(
        "--launcher",
        type=str,
        default="torchrun",
        choices=["torchrun", "accelerate", "python"],
    )
    parser.add_argument("--gpu-ids", type=str, default=None)
    parser.add_argument("--main-process-port", type=int, default=29500)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-train-epochs", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=None)
    parser.add_argument(
        "--generation-strategy",
        choices=["sample", "beam", "beam_sample"],
        default=None,
    )
    parser.add_argument("--diversity-penalty", type=float, default=None)
    parser.add_argument("--length-penalty", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--val-max-new-tokens", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--log-every-steps", type=int, default=None)
    parser.add_argument("--checkpoint-every-steps", type=int, default=None)
    parser.add_argument("--checkpoint-at-eval", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--keep-last-checkpoints", type=int, default=None)
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--mixed-precision", type=str, default=None)
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "bfloat16", "float32"], default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument(
        "--grpo-objective",
        choices=["group_relative_risk_kl", "clipped_grpo"],
        default=None,
    )
    parser.add_argument("--group-policy-scale", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--kl-coef", type=float, default=None)
    parser.add_argument("--score-batch-size", type=int, default=None)
    parser.add_argument("--disable-kl-monitor", action="store_true", default=None)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--load-in-4bit", action="store_true", default=None)
    parser.add_argument("--load-in-8bit", action="store_true", default=None)
    parser.add_argument("--torchao-int8-weight-only", action="store_true")
    parser.add_argument("--sft-per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--sft-per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--sft-gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--sft-learning-rate", type=float, default=None)
    parser.add_argument("--sft-num-train-epochs", type=float, default=None)
    parser.add_argument("--sft-mixed-precision", type=str, default=None)
    parser.add_argument("--sft-load-in-4bit", action="store_true")
    parser.add_argument("--sft-metric-max-new-tokens", type=int, default=None)
    parser.add_argument("--use-deepspeed-zero2", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use-deepspeed-zero3", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--zero3-init-flag", action="store_true", default=None)
    parser.add_argument("--zero3-save-16bit-model", action="store_true", default=None)
    parser.add_argument("--train-lora-module-filter", type=str, default=None)
    parser.add_argument("--train-projector", action="store_true", default=None)
    parser.add_argument("--projector-module-filter", type=str, default=None)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_repo_path(root_dir: Path, value: str | None) -> Path | None:
    if value in (None, "", "null"):
        return None
    path = Path(value)
    return path if path.is_absolute() else (root_dir / path)


def append_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            command.append(flag)
        return
    command.extend([flag, str(value)])


def append_bool_optional(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.append(flag if bool(value) else f"--no-{flag.removeprefix('--')}")


def pick_override(override: Any, default: Any) -> Any:
    return default if override is None else override


def resolve_run_suffix(args: argparse.Namespace) -> str | None:
    if args.run_suffix and args.auto_run_suffix:
        raise ValueError("--run-suffix and --auto-run-suffix cannot be used together.")
    if args.run_suffix:
        return args.run_suffix.strip().lstrip("_") or None
    if args.auto_run_suffix:
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    return None


def with_run_suffix(name: str, suffix: str | None) -> str:
    return name if not suffix else f"{name}_{suffix}"


def ensure_safe_output_dir(path: Path, args: argparse.Namespace, dry_run: bool) -> None:
    if dry_run or args.overwrite_existing:
        return
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite existing experiment directory: {path}. "
            "Use --run-suffix/--auto-run-suffix for a new run, or --overwrite-existing intentionally."
        )


def get_env_overrides(args: argparse.Namespace) -> dict[str, str] | None:
    if not args.gpu_ids:
        return None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    return env


def build_launch_prefix(args: argparse.Namespace, force_distributed: bool = False) -> list[str]:
    if args.num_processes <= 1 and not force_distributed:
        return [sys.executable]
    if args.launcher == "python":
        return [sys.executable]
    if args.launcher == "accelerate":
        return [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--num_processes",
            str(args.num_processes),
            "--main_process_port",
            str(args.main_process_port),
        ]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(args.num_processes),
        "--master_port",
        str(args.main_process_port),
    ]


def run_command(command: list[str], dry_run: bool, env: dict[str, str] | None = None) -> None:
    payload: dict[str, Any] = {"command": command}
    if env and env.get("CUDA_VISIBLE_DEVICES"):
        payload["cuda_visible_devices"] = env["CUDA_VISIBLE_DEVICES"]
    print(json.dumps(payload, ensure_ascii=False))
    if not dry_run:
        subprocess.run(command, check=True, env=env)


def write_runtime_files(
    experiment_dir: Path,
    resolved_config: dict[str, Any],
    commands: list[list[str]],
    dry_run: bool,
    args: argparse.Namespace,
) -> None:
    if dry_run:
        return
    ensure_safe_output_dir(experiment_dir, args, dry_run)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    (experiment_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved_config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (experiment_dir / "commands.txt").write_text(
        "\n".join(" ".join(command) for command in commands) + "\n",
        encoding="utf-8",
    )


def build_grpo_command(
    root_dir: Path,
    output_root: Path,
    common: dict[str, Any],
    experiment: dict[str, Any],
    sft_best_checkpoint: Path,
    base_model_path: str,
    args: argparse.Namespace,
) -> list[str]:
    grpo = common["grpo"]
    data = common["data"]
    weights = experiment["weights"]
    entity = {**grpo.get("entity", {}), **experiment.get("entity", {})}
    effective_use_zero2 = pick_override(args.use_deepspeed_zero2, grpo.get("use_deepspeed_zero2"))
    effective_use_zero3 = pick_override(args.use_deepspeed_zero3, grpo.get("use_deepspeed_zero3"))
    force_distributed = bool(effective_use_zero2 or effective_use_zero3)
    command = [
        *build_launch_prefix(args, force_distributed=force_distributed),
        str(root_dir / "scripts" / "train_gemma4_grpo_lora.py"),
    ]
    append_arg(command, "--base-model-path", base_model_path)
    append_arg(command, "--init-adapter-path", sft_best_checkpoint)
    append_arg(
        command,
        "--train-data-path",
        args.train_data_path or resolve_repo_path(root_dir, data.get("grpo_train_data_path", data["train_data_path"])),
    )
    append_arg(command, "--train-entity-path", args.train_entity_path or resolve_repo_path(root_dir, data.get("train_entity_path")))
    append_arg(command, "--val-data-path", args.val_data_path or resolve_repo_path(root_dir, data["val_data_path"]))
    append_arg(command, "--val-entity-path", args.val_entity_path or resolve_repo_path(root_dir, data.get("val_entity_path")))
    append_arg(command, "--audio-prefix-from", pick_override(args.audio_prefix_from, data.get("audio_prefix_from")))
    append_arg(command, "--audio-prefix-to", pick_override(args.audio_prefix_to, data.get("audio_prefix_to")))
    append_arg(command, "--output-root", output_root)
    append_arg(command, "--experiment-name", experiment["experiment_name"])
    append_arg(
        command,
        "--grpo-objective",
        pick_override(
            args.grpo_objective,
            experiment.get("grpo_objective", grpo.get("grpo_objective", "group_relative_risk_kl")),
        ),
    )
    append_arg(command, "--objective", "weighted_sum")
    append_arg(command, "--bleu-weight", weights["bleu"])
    append_arg(command, "--chrf-weight", weights["chrf"])
    append_arg(command, "--key-weight", weights.get("entity", weights.get("key", 0.0)))
    append_arg(command, "--ce-weight", weights["ce"])
    overrides = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "num_candidates": args.num_candidates,
        "generation_strategy": args.generation_strategy,
        "diversity_penalty": args.diversity_penalty,
        "length_penalty": args.length_penalty,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "max_new_tokens": args.max_new_tokens,
        "val_max_new_tokens": args.val_max_new_tokens,
        "mixed_precision": args.mixed_precision,
        "torch_dtype": args.torch_dtype,
        "warmup_ratio": args.warmup_ratio,
        "group_policy_scale": args.group_policy_scale,
        "clip_range": args.clip_range,
        "kl_coef": args.kl_coef,
        "score_batch_size": args.score_batch_size,
        "gradient_checkpointing": args.gradient_checkpointing,
        "eval_every_steps": args.eval_every_steps,
        "log_every_steps": args.log_every_steps,
        "key_match_mode": args.key_match_mode,
        "entity_reward_mode": args.entity_reward_mode,
        "entity_soft_tau": args.entity_soft_tau,
        "entity_embedding_model": args.entity_embedding_model,
        "entity_embedding_max_length": args.entity_embedding_max_length,
        "entity_embedding_pooling": args.entity_embedding_pooling,
        "entity_embedding_device": args.entity_embedding_device,
        "entity_extractor_device": args.entity_extractor_device,
        "ner_tokenizer_model": args.ner_tokenizer_model,
        "ner_model": args.ner_model,
        "ner_tokenizer_path": args.ner_tokenizer_path,
        "ner_path": args.ner_path,
        "train_lora_module_filter": args.train_lora_module_filter,
        "projector_module_filter": args.projector_module_filter,
        "checkpoint_every_steps": args.checkpoint_every_steps,
        "checkpoint_at_eval": args.checkpoint_at_eval,
        "keep_last_checkpoints": args.keep_last_checkpoints,
    }
    for key, flag in (
        ("per_device_train_batch_size", "--per-device-train-batch-size"),
        ("eval_batch_size", "--eval-batch-size"),
        ("gradient_accumulation_steps", "--gradient-accumulation-steps"),
        ("learning_rate", "--learning-rate"),
        ("num_train_epochs", "--num-train-epochs"),
        ("num_candidates", "--num-candidates"),
        ("generation_strategy", "--generation-strategy"),
        ("diversity_penalty", "--diversity-penalty"),
        ("length_penalty", "--length-penalty"),
        ("temperature", "--temperature"),
        ("top_p", "--top-p"),
        ("top_k", "--top-k"),
        ("repetition_penalty", "--repetition-penalty"),
        ("no_repeat_ngram_size", "--no-repeat-ngram-size"),
        ("max_new_tokens", "--max-new-tokens"),
        ("val_max_new_tokens", "--val-max-new-tokens"),
        ("mixed_precision", "--mixed-precision"),
        ("group_policy_scale", "--group-policy-scale"),
        ("clip_range", "--clip-range"),
        ("kl_coef", "--kl-coef"),
        ("score_batch_size", "--score-batch-size"),
        ("eval_every_steps", "--eval-every-steps"),
        ("log_every_steps", "--log-every-steps"),
        ("checkpoint_every_steps", "--checkpoint-every-steps"),
        ("keep_last_checkpoints", "--keep-last-checkpoints"),
        ("key_match_mode", "--key-match-mode"),
        ("entity_reward_mode", "--entity-reward-mode"),
        ("entity_soft_tau", "--entity-soft-tau"),
        ("entity_embedding_model", "--entity-embedding-model"),
        ("entity_embedding_max_length", "--entity-embedding-max-length"),
        ("entity_embedding_pooling", "--entity-embedding-pooling"),
        ("entity_embedding_device", "--entity-embedding-device"),
        ("entity_extractor_device", "--entity-extractor-device"),
        ("ner_tokenizer_model", "--ner-tokenizer-model"),
        ("ner_model", "--ner-model"),
        ("ner_tokenizer_path", "--ner-tokenizer-path"),
        ("ner_path", "--ner-path"),
        ("warmup_ratio", "--warmup-ratio"),
        ("weight_decay", "--weight-decay"),
        ("seed", "--seed"),
        ("torch_dtype", "--torch-dtype"),
        ("attn_implementation", "--attn-implementation"),
        ("sampling_rate", "--sampling-rate"),
        ("lora_r", "--lora-r"),
        ("lora_alpha", "--lora-alpha"),
        ("lora_dropout", "--lora-dropout"),
        ("lora_target_modules", "--lora-target-modules"),
        ("train_lora_module_filter", "--train-lora-module-filter"),
        ("projector_module_filter", "--projector-module-filter"),
    ):
        append_arg(
            command,
            flag,
            pick_override(overrides.get(key), experiment.get(key, entity.get(key, grpo.get(key)))),
        )
    append_arg(command, "--enable-tensorboard", grpo.get("enable_tensorboard"))
    append_arg(command, "--gradient-checkpointing", pick_override(args.gradient_checkpointing, grpo.get("gradient_checkpointing")))
    append_arg(command, "--load-in-4bit", pick_override(args.load_in_4bit, grpo.get("load_in_4bit")))
    append_arg(command, "--load-in-8bit", pick_override(args.load_in_8bit, grpo.get("load_in_8bit")))
    append_arg(command, "--disable-kl-monitor", pick_override(args.disable_kl_monitor, grpo.get("disable_kl_monitor")))
    append_bool_optional(command, "--checkpoint-at-eval", pick_override(args.checkpoint_at_eval, grpo.get("checkpoint_at_eval")))
    append_bool_optional(
        command,
        "--entity-include-per",
        pick_override(
            args.entity_include_per,
            experiment.get("entity_include_per", entity.get("entity_include_per", grpo.get("entity_include_per"))),
        ),
    )
    append_arg(command, "--resume-from-checkpoint", args.resume_from_checkpoint)
    append_arg(command, "--auto-resume", args.auto_resume or grpo.get("auto_resume"))
    append_arg(command, "--torchao-int8-weight-only", args.torchao_int8_weight_only or grpo.get("torchao_int8_weight_only"))
    append_arg(command, "--use-deepspeed-zero2", effective_use_zero2)
    append_arg(command, "--use-deepspeed-zero3", effective_use_zero3)
    append_arg(command, "--zero3-init-flag", pick_override(args.zero3_init_flag, grpo.get("zero3_init_flag")))
    append_arg(command, "--zero3-save-16bit-model", pick_override(args.zero3_save_16bit_model, grpo.get("zero3_save_16bit_model")))
    append_arg(command, "--train-projector", args.train_projector or grpo.get("train_projector"))
    effective_load_in_4bit = pick_override(args.load_in_4bit, grpo.get("load_in_4bit"))
    if effective_load_in_4bit:
        append_arg(command, "--bnb-4bit-use-double-quant", grpo.get("bnb_4bit_use_double_quant"))
        append_arg(command, "--bnb-4bit-quant-type", grpo.get("bnb_4bit_quant_type"))
        append_arg(command, "--bnb-4bit-compute-dtype", grpo.get("bnb_4bit_compute_dtype"))
    return command


def build_eval_command(
    root_dir: Path,
    experiment_dir: Path,
    common: dict[str, Any],
    base_model_path: str,
    adapter_path: Path,
    experiment_name: str,
    args: argparse.Namespace,
    weights: dict[str, Any],
) -> list[str]:
    data = common["data"]
    grpo = common["grpo"]
    command = [sys.executable, str(root_dir / "scripts" / "eval_testt_bleu_chrf.py")]
    append_arg(command, "--base-model-path", base_model_path)
    append_arg(command, "--adapter-path", adapter_path)
    append_arg(command, "--data-path", args.test_data_path or resolve_repo_path(root_dir, data["test_data_path"]))
    append_arg(command, "--entity-path", args.test_entity_path or resolve_repo_path(root_dir, data.get("test_entity_path")))
    append_arg(command, "--audio-prefix-from", pick_override(args.audio_prefix_from, data.get("audio_prefix_from")))
    append_arg(command, "--audio-prefix-to", pick_override(args.audio_prefix_to, data.get("audio_prefix_to")))
    append_arg(command, "--output-dir", experiment_dir / "eval")
    append_arg(command, "--experiment-name", experiment_name)
    append_arg(command, "--objective", "weighted_sum")
    append_arg(command, "--bleu-weight", weights["bleu"])
    append_arg(command, "--chrf-weight", weights["chrf"])
    append_arg(command, "--key-weight", weights.get("entity", weights.get("key", 0.0)))
    append_arg(command, "--ce-weight", weights["ce"])
    append_arg(command, "--batch-size", pick_override(args.eval_batch_size, grpo.get("eval_batch_size")))
    append_arg(command, "--max-new-tokens", grpo.get("max_new_tokens"))
    append_arg(command, "--sampling-rate", grpo.get("sampling_rate"))
    append_arg(command, "--torch-dtype", grpo.get("torch_dtype"))
    append_arg(command, "--attn-implementation", grpo.get("attn_implementation"))
    append_arg(command, "--key-match-mode", pick_override(args.key_match_mode, grpo.get("key_match_mode")))
    return command


def build_sft_command(
    root_dir: Path,
    output_root: Path,
    common: dict[str, Any],
    base_model_path: str,
    args: argparse.Namespace,
) -> tuple[list[str], Path, str]:
    sft = common["sft"]
    data = common["data"]
    experiment_name = with_run_suffix(
        args.sft_experiment_name or sft["experiment_name"],
        args.resolved_run_suffix,
    )
    command = [*build_launch_prefix(args), str(root_dir / "scripts" / "train_gemma4_sft_qlora.py")]
    append_arg(command, "--base-model-path", base_model_path)
    append_arg(
        command,
        "--train-data-path",
        args.train_data_path or resolve_repo_path(root_dir, data.get("sft_train_data_path", data["train_data_path"])),
    )
    append_arg(command, "--val-data-path", args.val_data_path or resolve_repo_path(root_dir, data["val_data_path"]))
    append_arg(command, "--test-data-path", args.test_data_path or resolve_repo_path(root_dir, data["test_data_path"]))
    append_arg(command, "--audio-prefix-from", pick_override(args.audio_prefix_from, data.get("audio_prefix_from")))
    append_arg(command, "--audio-prefix-to", pick_override(args.audio_prefix_to, data.get("audio_prefix_to")))
    append_arg(command, "--output-root", output_root)
    append_arg(command, "--experiment-name", experiment_name)
    sft_overrides = {
        "per_device_train_batch_size": args.sft_per_device_train_batch_size,
        "per_device_eval_batch_size": args.sft_per_device_eval_batch_size,
        "gradient_accumulation_steps": args.sft_gradient_accumulation_steps,
        "learning_rate": args.sft_learning_rate,
        "num_train_epochs": args.sft_num_train_epochs,
        "mixed_precision": args.sft_mixed_precision,
        "load_in_4bit": args.sft_load_in_4bit,
        "metric_max_new_tokens": args.sft_metric_max_new_tokens,
    }
    for key, flag in (
        ("per_device_train_batch_size", "--per-device-train-batch-size"),
        ("per_device_eval_batch_size", "--per-device-eval-batch-size"),
        ("gradient_accumulation_steps", "--gradient-accumulation-steps"),
        ("learning_rate", "--learning-rate"),
        ("num_train_epochs", "--num-train-epochs"),
        ("mixed_precision", "--mixed-precision"),
        ("load_in_4bit", "--load-in-4bit"),
        ("gradient_checkpointing", "--gradient-checkpointing"),
        ("enable_tensorboard", "--enable-tensorboard"),
        ("seed", "--seed"),
        ("save_steps", "--save-steps"),
        ("eval_steps", "--eval-steps"),
        ("logging_steps", "--logging-steps"),
        ("save_total_limit", "--save-total-limit"),
        ("metric_max_new_tokens", "--metric-max-new-tokens"),
        ("torch_dtype", "--torch-dtype"),
        ("attn_implementation", "--attn-implementation"),
        ("optim", "--optim"),
        ("lora_r", "--lora-r"),
        ("lora_alpha", "--lora-alpha"),
        ("lora_dropout", "--lora-dropout"),
        ("lora_target_modules", "--lora-target-modules"),
    ):
        append_arg(command, flag, pick_override(sft_overrides.get(key), sft.get(key)))
    append_arg(command, "--device-map", sft.get("device_map"))
    return command, output_root / experiment_name / "adapter_best", experiment_name


def load_common_and_experiment(
    config_root: Path,
    experiment_name: str | None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    common = load_yaml(config_root / "common.yaml")
    experiment = None
    if experiment_name is not None:
        experiment = load_yaml(config_root / f"{experiment_name}.yaml")
    return common, experiment


def execute_experiment(
    root_dir: Path,
    output_root: Path,
    common: dict[str, Any],
    experiment: dict[str, Any],
    sft_best_checkpoint: Path,
    base_model_path: str,
    dry_run: bool,
    args: argparse.Namespace,
) -> None:
    experiment = copy.deepcopy(experiment)
    experiment["experiment_name"] = with_run_suffix(
        experiment["experiment_name"],
        args.resolved_run_suffix,
    )
    experiment_name = experiment["experiment_name"]
    experiment_dir = output_root / experiment_name
    ensure_safe_output_dir(experiment_dir, args, dry_run)
    train_command = build_grpo_command(
        root_dir=root_dir,
        output_root=output_root,
        common=common,
        experiment=experiment,
        sft_best_checkpoint=sft_best_checkpoint,
        base_model_path=base_model_path,
        args=args,
    )
    eval_command = build_eval_command(
        root_dir=root_dir,
        experiment_dir=experiment_dir,
        common=common,
        base_model_path=base_model_path,
        adapter_path=experiment_dir / "adapter_best",
        experiment_name=experiment_name,
        args=args,
        weights=experiment["weights"],
    )
    resolved_config = {
        "experiment": experiment,
        "common": common,
        "base_model_path": base_model_path,
        "output_root": str(output_root),
        "sft_best_checkpoint": str(sft_best_checkpoint),
        "data_overrides": {
            "train_data_path": str(args.train_data_path) if args.train_data_path else None,
            "train_entity_path": str(args.train_entity_path) if args.train_entity_path else None,
            "val_data_path": str(args.val_data_path) if args.val_data_path else None,
            "val_entity_path": str(args.val_entity_path) if args.val_entity_path else None,
            "test_data_path": str(args.test_data_path) if args.test_data_path else None,
            "test_entity_path": str(args.test_entity_path) if args.test_entity_path else None,
            "audio_prefix_from": args.audio_prefix_from,
            "audio_prefix_to": args.audio_prefix_to,
        },
    }
    write_runtime_files(experiment_dir, resolved_config, [train_command, eval_command], dry_run, args)
    env = get_env_overrides(args)
    run_command(train_command, dry_run, env=env)
    run_command(eval_command, dry_run, env=env)


def main() -> None:
    args = parse_args()
    args.resolved_run_suffix = resolve_run_suffix(args)
    root_dir = detect_repo_root()
    common, experiment = load_common_and_experiment(args.config_root, args.experiment)
    base_model_path = args.base_model_path or common["runtime"]["base_model_path"]
    dry_run = args.command in {"dry-run", "dry-run-sft"}

    if args.command in {"run-sft", "dry-run-sft"}:
        output_root = args.output_root or resolve_repo_path(
            root_dir,
            common["runtime"].get("sft_output_root", common["runtime"]["output_root"]),
        )
        sft_command, adapter_best_path, experiment_name = build_sft_command(
            root_dir=root_dir,
            output_root=output_root,
            common=common,
            base_model_path=base_model_path,
            args=args,
        )
        ensure_safe_output_dir(output_root / experiment_name, args, dry_run)
        eval_command = build_eval_command(
            root_dir=root_dir,
            experiment_dir=output_root / experiment_name,
            common=common,
            base_model_path=base_model_path,
            adapter_path=adapter_best_path,
            experiment_name=experiment_name,
            args=args,
            weights={"bleu": 0.0, "chrf": 0.0, "key": 0.0, "ce": 1.0},
        )
        resolved_config = {
            "common": common,
            "base_model_path": base_model_path,
            "output_root": str(output_root),
            "sft_experiment_name": experiment_name,
            "sft_best_checkpoint": str(adapter_best_path),
            "data_overrides": {
                "train_data_path": str(args.train_data_path) if args.train_data_path else None,
                "val_data_path": str(args.val_data_path) if args.val_data_path else None,
                "test_data_path": str(args.test_data_path) if args.test_data_path else None,
                "audio_prefix_from": args.audio_prefix_from,
                "audio_prefix_to": args.audio_prefix_to,
            },
        }
        write_runtime_files(output_root / experiment_name, resolved_config, [sft_command, eval_command], dry_run, args)
        env = get_env_overrides(args)
        run_command(sft_command, dry_run, env=env)
        run_command(eval_command, dry_run, env=env)
        return

    output_root = args.output_root or (root_dir / common["runtime"]["output_root"])

    if args.command in {"run-one", "dry-run"}:
        if experiment is None:
            raise ValueError("--experiment is required for run-one/dry-run.")
        if args.sft_best_checkpoint is None:
            raise ValueError("--sft-best-checkpoint is required for run-one/dry-run.")
        execute_experiment(
            root_dir=root_dir,
            output_root=output_root,
            common=common,
            experiment=experiment,
            sft_best_checkpoint=args.sft_best_checkpoint,
            base_model_path=base_model_path,
            dry_run=dry_run,
            args=args,
        )
        return

    if args.command == "run-all":
        if args.sft_best_checkpoint is None:
            raise ValueError("--sft-best-checkpoint is required for run-all.")
        for experiment_name in EXPERIMENT_ORDER:
            _, experiment = load_common_and_experiment(args.config_root, experiment_name)
            execute_experiment(
                root_dir=root_dir,
                output_root=output_root,
                common=common,
                experiment=experiment,
                sft_best_checkpoint=args.sft_best_checkpoint,
                base_model_path=base_model_path,
                dry_run=False,
                args=args,
            )


if __name__ == "__main__":
    main()

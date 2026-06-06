# -----------------------------------------------------------------------------
# Third-party baseline (best-effort, for paper comparison only).
# This script wires up an external, off-the-shelf model to reproduce one of the
# baseline rows reported in the EARL paper. It depends on third-party model
# weights/APIs that are NOT part of EARL and may break with upstream changes.
# It is not required to train or evaluate EARL itself; the core SFT + GRPO
# pipeline and the entity-recall metric live in the parent scripts/ directory.
# -----------------------------------------------------------------------------
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


DEFAULT_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)
DEFAULT_DATA_PROMPT = (
    "Translate the following speech segment into chinese. Follow these specific "
    "instructions for formatting the answer:\n"
    "* Only output the translation, with no newlines.\n"
    "* When translating numbers, write the digits, i.e. write 1.7 and not one "
    "point seven, and write 3 instead of three."
)
DEFAULT_LORA_TARGET_MODULES = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="LoRA SFT for Qwen2.5-Omni on the compact Kyrgyz speech -> Chinese dataset."
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=str(root_dir / "model" / "Qwen2.5-Omni-3B"),
        help="Local Qwen2.5-Omni-3B directory or Hugging Face model id.",
    )
    parser.add_argument(
        "--train-data-path",
        type=Path,
        default=root_dir / "data" / "train_ky2zh_with_ky_text.jsonl",
        help="Compact JSONL with id/audio/prompt/source_text/text_zh fields.",
    )
    parser.add_argument("--output-root", type=Path, default=root_dir / "model" / "sft")
    parser.add_argument("--experiment-name", type=str, default="sft_qwen25_omni_3b_ky2zh_lora")
    parser.add_argument("--audio-prefix-from", type=str, default="")
    parser.add_argument("--audio-prefix-to", type=str, default="")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-train-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        default="cosine",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument(
        "--optim",
        choices=["paged_adamw_8bit", "adamw_torch"],
        default="paged_adamw_8bit",
    )
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--enable-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bnb-4bit-quant-type", choices=["nf4", "fp4"], default="nf4")
    parser.add_argument("--bnb-4bit-compute-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", action="store_false", dest="use_lora")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument(
        "--use-dataset-prompt",
        action="store_true",
        default=True,
        help="Use the prompt saved in the compact JSONL.",
    )
    parser.add_argument("--prompt-text", type=str, default=DEFAULT_DATA_PROMPT)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bind_local_cuda_device_from_env() -> None:
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank >= 0:
            torch.cuda.set_device(local_rank)


def resolve_torch_dtype(dtype_name: str) -> str | torch.dtype:
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def resolve_explicit_torch_dtype(dtype_name: str) -> torch.dtype:
    return getattr(torch, dtype_name)


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def rewrite_path(path: str, prefix_from: str, prefix_to: str) -> str:
    if not prefix_from:
        return path
    if path.startswith(prefix_to):
        return path
    if not path.startswith(prefix_from):
        raise ValueError(f"Path {path!r} does not start with {prefix_from!r}.")
    return prefix_to + path.removeprefix(prefix_from)


def load_audio(path: str, sampling_rate: int):
    try:
        import librosa
    except ModuleNotFoundError as exc:
        raise RuntimeError("Qwen2.5-Omni SFT requires librosa to load audio.") from exc
    audio, _ = librosa.load(path, sr=sampling_rate, mono=True)
    return audio


def load_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with args.train_data_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            audio = str(record.get("audio", "") or "")
            target = normalize_text(record.get("text_zh", ""))
            if not audio or not target:
                continue
            dataset_prompt = str(record.get("prompt", "") or "").strip()
            rows.append(
                {
                    "id": str(record.get("id", f"line_{line_number}")),
                    "audio": rewrite_path(audio, args.audio_prefix_from, args.audio_prefix_to),
                    "prompt": dataset_prompt if args.use_dataset_prompt and dataset_prompt else args.prompt_text,
                    "reference": target,
                    "source_text": normalize_text(record.get("source_text", "")),
                }
            )
    if not rows:
        raise ValueError(f"No usable rows loaded from {args.train_data_path}.")
    return rows


def split_rows(
    rows: list[dict[str, str]],
    val_ratio: float,
    seed: int,
    limit_train: int | None,
    limit_val: int | None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not (0.0 < val_ratio < 0.5):
        raise ValueError("--val-ratio must be between 0 and 0.5.")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    val_count = max(1, int(round(len(shuffled) * val_ratio)))
    val_rows = shuffled[:val_count]
    train_rows = shuffled[val_count:]
    if limit_train is not None:
        train_rows = train_rows[:limit_train]
    if limit_val is not None:
        val_rows = val_rows[:limit_val]
    return train_rows, val_rows


class RowDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.rows[idx]


def build_prompt_messages(row: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": row["audio"]},
                {"type": "text", "text": row["prompt"]},
            ],
        },
    ]


def build_training_messages(row: dict[str, str]) -> list[dict[str, Any]]:
    return build_prompt_messages(row) + [
        {"role": "assistant", "content": [{"type": "text", "text": row["reference"]}]}
    ]


@dataclass
class QwenOmniSftCollator:
    processor: Any
    sampling_rate: int

    def _encode(self, conversations: list[list[dict[str, Any]]], audios: list[Any]) -> dict[str, torch.Tensor]:
        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=False,
            tokenize=False,
        )
        return self.processor(
            text=text,
            audio=audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )

    def __call__(self, features: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        audios = [load_audio(row["audio"], self.sampling_rate) for row in features]
        prompt_text = self.processor.apply_chat_template(
            [build_prompt_messages(row) for row in features],
            add_generation_prompt=True,
            tokenize=False,
        )
        full_inputs = self._encode([build_training_messages(row) for row in features], audios)
        prompt_inputs = self.processor(
            text=prompt_text,
            audio=audios,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        labels = full_inputs["input_ids"].clone()
        attention_mask = full_inputs["attention_mask"]
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()
        for row_index, prompt_length in enumerate(prompt_lengths):
            labels[row_index, : min(int(prompt_length), labels.shape[1])] = -100
        labels = labels.masked_fill(attention_mask == 0, -100)
        full_inputs["labels"] = labels
        return full_inputs


class JsonlLoggingCallback(TrainerCallback):
    def __init__(self, path: Path) -> None:
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs or not state.is_local_process_zero:
            return
        row = {"step": state.global_step, "epoch": state.epoch}
        row.update(logs)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class LastCheckpointTrackerCallback(TrainerCallback):
    def __init__(self) -> None:
        self.last_checkpoint_dir: Path | None = None

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            self.last_checkpoint_dir = checkpoint_dir


def load_qwen_omni(args: argparse.Namespace):
    try:
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        model_class = Qwen2_5OmniThinkerForConditionalGeneration
        full_omni = False
    except ImportError:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        model_class = Qwen2_5OmniForConditionalGeneration
        full_omni = True

    processor = Qwen2_5OmniProcessor.from_pretrained(args.base_model_path)
    model_kwargs: dict[str, Any] = {
        "dtype": resolve_torch_dtype(args.torch_dtype),
        "low_cpu_mem_usage": True,
    }
    if full_omni:
        model_kwargs["enable_audio_output"] = False
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=resolve_explicit_torch_dtype(args.bnb_4bit_compute_dtype),
        )
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if torch.cuda.is_available() and local_rank >= 0:
        model_kwargs["device_map"] = {"": local_rank}
    elif args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    try:
        model = model_class.from_pretrained(args.base_model_path, **model_kwargs)
    except TypeError:
        dtype = model_kwargs.pop("dtype", None)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        model = model_class.from_pretrained(args.base_model_path, **model_kwargs)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return processor, model


def resolve_lora_target_modules(model: torch.nn.Module, requested: str) -> list[str]:
    suffixes = [item.strip() for item in requested.split(",") if item.strip()]
    resolved: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(name == suffix or name.endswith(f".{suffix}") for suffix in suffixes):
            resolved.append(name)
    resolved = sorted(set(resolved))
    if not resolved:
        raise ValueError(f"No LoRA target modules matched: {suffixes}")
    return resolved


def apply_lora(model: torch.nn.Module, args: argparse.Namespace):
    if not args.use_lora:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return model, [], trainable, total
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ModuleNotFoundError as exc:
        raise RuntimeError("peft is required for LoRA SFT training.") from exc

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    target_modules = resolve_lora_target_modules(model, args.lora_target_modules)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, target_modules, trainable, total


def build_training_arguments(args: argparse.Namespace, trainer_output_dir: Path, tensorboard_dir: Path) -> TrainingArguments:
    use_steps = args.eval_steps > 0 or args.save_steps > 0
    eval_strategy = "steps" if use_steps else "epoch"
    save_strategy = "steps" if use_steps else "epoch"
    effective_eval_steps = args.eval_steps if args.eval_steps > 0 else None
    effective_save_steps = args.save_steps if args.save_steps > 0 else None
    if use_steps:
        interval = effective_eval_steps or effective_save_steps or 100
        effective_eval_steps = interval
        effective_save_steps = interval
    report_to = ["tensorboard"] if args.enable_tensorboard else []
    signature = inspect.signature(TrainingArguments.__init__)
    kwargs: dict[str, Any] = {
        "output_dir": str(trainer_output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_train_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "save_strategy": save_strategy,
        "eval_steps": effective_eval_steps,
        "save_steps": effective_save_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": args.mixed_precision == "bf16",
        "fp16": args.mixed_precision == "fp16",
        "optim": args.optim,
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "report_to": report_to,
        "logging_dir": str(tensorboard_dir),
        "dataloader_pin_memory": True,
        "disable_tqdm": False,
    }
    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = eval_strategy
    elif "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = eval_strategy
    return TrainingArguments(**kwargs)


def copy_adapter_artifacts(source_dir: Path, target_dir: Path) -> bool:
    candidate_files = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin", "README.md"]
    available = [source_dir / name for name in candidate_files if (source_dir / name).exists()]
    if not available:
        return False
    staging = target_dir.with_name(f"{target_dir.name}.tmp")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    for source in available:
        shutil.copyfile(source, staging / source.name)
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    staging.replace(target_dir)
    return True


def save_model_or_adapter(model: torch.nn.Module, output_dir: Path) -> None:
    staging = output_dir.with_name(f"{output_dir.name}.tmp")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(staging)
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    staging.replace(output_dir)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    bind_local_cuda_device_from_env()
    seed_everything(args.seed)

    all_rows = load_rows(args)
    train_rows, val_rows = split_rows(
        all_rows,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit_train=args.limit_train_samples,
        limit_val=args.limit_val_samples,
    )

    experiment_dir = args.output_root / args.experiment_name
    adapter_best_dir = experiment_dir / "adapter_best"
    adapter_last_dir = experiment_dir / "adapter_last"
    model_best_dir = experiment_dir / "model_best"
    trainer_output_dir = experiment_dir / "trainer_output"
    tensorboard_dir = args.tensorboard_dir or (experiment_dir / "tensorboard")
    trainer_logs_path = experiment_dir / "trainer_logs.jsonl"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    processor, model = load_qwen_omni(args)
    model, target_modules, trainable_params, total_params = apply_lora(model, args)
    collator = QwenOmniSftCollator(processor=processor, sampling_rate=args.sampling_rate)
    training_args = build_training_arguments(args, trainer_output_dir, tensorboard_dir)

    world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
    global_micro_batch = args.per_device_train_batch_size * world_size
    effective_batch = global_micro_batch * args.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(train_rows) / max(global_micro_batch, 1))
    optimizer_steps_per_epoch = math.ceil(steps_per_epoch / max(args.gradient_accumulation_steps, 1))

    run_config = {
        "experiment_name": args.experiment_name,
        "base_model_path": args.base_model_path,
        "train_data_path": str(args.train_data_path),
        "output_dir": str(experiment_dir),
        "audio_prefix_from": args.audio_prefix_from,
        "audio_prefix_to": args.audio_prefix_to,
        "train_samples": len(train_rows),
        "val_samples": len(val_rows),
        "val_ratio": args.val_ratio,
        "sampling_rate": args.sampling_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_global_batch_size": effective_batch,
        "num_train_epochs": args.num_train_epochs,
        "max_train_steps": args.max_train_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "mixed_precision": args.mixed_precision,
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
        "gradient_checkpointing": args.gradient_checkpointing,
        "load_in_4bit": args.load_in_4bit,
        "use_lora": args.use_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": target_modules,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "world_size": world_size,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
    }
    if int(os.environ.get("RANK", "0")) == 0:
        save_json(experiment_dir / "run_config.json", run_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=RowDataset(train_rows),
        eval_dataset=RowDataset(val_rows),
        data_collator=collator,
        callbacks=[JsonlLoggingCallback(trainer_logs_path)],
    )
    checkpoint_tracker = LastCheckpointTrackerCallback()
    trainer.add_callback(checkpoint_tracker)

    if trainer.is_world_process_zero():
        print(
            json.dumps(
                {
                    "experiment_name": args.experiment_name,
                    "train_samples": len(train_rows),
                    "val_samples": len(val_rows),
                    "effective_global_batch_size": effective_batch,
                    "target_modules": target_modules,
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                    "trainer_output_dir": str(trainer_output_dir),
                },
                ensure_ascii=False,
            )
        )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        trainer.state.save_to_json(str(experiment_dir / "train_state.json"))
    final_eval_metrics = trainer.evaluate()
    trainer.accelerator.wait_for_everyone()

    last_checkpoint = (
        str(checkpoint_tracker.last_checkpoint_dir)
        if checkpoint_tracker.last_checkpoint_dir is not None
        else get_last_checkpoint(str(trainer_output_dir))
    )
    copied_last = False
    copied_best = False
    if trainer.is_world_process_zero():
        if args.use_lora:
            if last_checkpoint:
                copied_last = copy_adapter_artifacts(Path(last_checkpoint), adapter_last_dir)
            if trainer.state.best_model_checkpoint:
                copied_best = copy_adapter_artifacts(Path(trainer.state.best_model_checkpoint), adapter_best_dir)
            if not copied_best and copied_last:
                copied_best = copy_adapter_artifacts(adapter_last_dir, adapter_best_dir)
            if not copied_best:
                save_model_or_adapter(trainer.model, adapter_best_dir)
        else:
            save_model_or_adapter(trainer.model, model_best_dir)

    metrics = dict(train_result.metrics)
    metrics.update({f"final_{k}": v for k, v in final_eval_metrics.items()})
    metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint
    metrics["last_checkpoint"] = last_checkpoint
    metrics["adapter_last_available"] = copied_last
    if trainer.is_world_process_zero():
        save_json(experiment_dir / "train_result.json", metrics)
        print(
            json.dumps(
                {
                    "adapter_best": str(adapter_best_dir) if args.use_lora else None,
                    "adapter_last": str(adapter_last_dir) if copied_last else None,
                    "model_best": str(model_best_dir) if not args.use_lora else None,
                    "run_config": str(experiment_dir / "run_config.json"),
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()

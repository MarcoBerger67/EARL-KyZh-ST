from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import get_scheduler

from fca_grpo_risk_runtime import (
    DEFAULT_BASE_MODEL_PATH,
    DEFAULT_LORA_TARGET_MODULES,
    append_jsonl,
    apply_lora,
    align_audio_feature_mask,
    attach_sidecar_annotations,
    attach_sidecar_keys,
    bind_local_cuda_device_from_env,
    build_entity_rewarder,
    build_accelerator,
    build_prompt_messages,
    collect_quantization_summary,
    compute_candidate_reward,
    evaluate_validation,
    gemma4_zero3_gathered_forward_context,
    get_audio_token_id,
    get_pad_token_id,
    get_runtime_float_dtype,
    get_summary_writer,
    get_vocab_size,
    iter_batches,
    load_entity_sidecar,
    load_key_sidecar,
    load_model_and_processor,
    load_samples,
    move_batch_to_device,
    patch_gemma4_zero3_forward,
    patch_torch_finfo_for_quantized_gemma4,
    patch_torch_masked_fill_for_low_precision,
    patch_torch_masked_scatter_for_quantized_gemma4,
    save_json,
    seed_everything,
    should_enable_model_gradient_checkpointing,
    trim_surplus_audio_tokens,
)


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train Gemma4 with LoRA-based GRPO from an SFT checkpoint."
    )
    parser.add_argument("--base-model-path", type=str, default=DEFAULT_BASE_MODEL_PATH)
    parser.add_argument("--init-adapter-path", type=Path, required=True)
    parser.add_argument(
        "--train-data-path",
        type=Path,
        default=(
            root_dir
            / "data"
            / "converted_testt_format"
            / "train_ky2zh_full285h_stage1.cleaned.jsonl"
        ),
    )
    parser.add_argument(
        "--val-data-path",
        type=Path,
        default=root_dir / "data" / "converted_testt_format" / "val_ky2zh_full285h.jsonl",
    )
    parser.add_argument("--train-entity-path", type=Path, default=None)
    parser.add_argument("--val-entity-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=root_dir / "model" / "grpo")
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--enable-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument(
        "--objective",
        choices=["weighted_sum", "bleu_only", "chrf_only"],
        default="weighted_sum",
    )
    parser.add_argument("--bleu-weight", type=float, default=0.0)
    parser.add_argument("--chrf-weight", type=float, default=0.0)
    parser.add_argument("--entity-weight", dest="key_weight", type=float, default=None)
    parser.add_argument("--key-weight", type=float, default=0.0)
    parser.add_argument("--ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--entity-reward-mode",
        choices=["none", "key_recall", "entity_em", "entity_soft", "entity_gemma_fuzzy", "entity_substring"],
        default="key_recall",
    )
    parser.add_argument("--entity-soft-tau", type=float, default=0.6)
    parser.add_argument("--entity-embedding-model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--entity-embedding-max-length", type=int, default=128)
    parser.add_argument("--entity-embedding-pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--entity-embedding-device", type=str, default="cpu")
    parser.add_argument("--entity-extractor-device", type=str, default="cpu")
    parser.add_argument("--ner-tokenizer-model", type=str, default="FINE_ELECTRA_SMALL_ZH")
    parser.add_argument("--ner-model", type=str, default="MSRA_NER_ELECTRA_SMALL_ZH")
    parser.add_argument("--ner-tokenizer-path", type=str, default=None)
    parser.add_argument("--ner-path", type=str, default=None)
    parser.add_argument("--audio-prefix-from", type=str, default="")
    parser.add_argument("--audio-prefix-to", type=str, default="")
    parser.add_argument("--num-train-epochs", type=lambda x: int(float(x)), default=1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        default="cosine",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-candidates", type=int, default=4)
    parser.add_argument(
        "--generation-strategy",
        choices=["sample", "beam", "beam_sample"],
        default="beam_sample",
    )
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--val-max-new-tokens", type=int, default=None)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--kl-coef", type=float, default=0.02)
    parser.add_argument("--score-batch-size", type=int, default=2)
    parser.add_argument("--group-policy-scale", type=float, default=None)
    parser.add_argument("--disable-kl-monitor", action="store_true")
    parser.add_argument(
        "--key-match-mode",
        type=str,
        default="normalized_exact",
        choices=("exact", "normalized_exact", "fuzzy_substring"),
    )
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
    )
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument(
        "--bnb-4bit-quant-type",
        choices=["nf4", "fp4"],
        default="nf4",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--bnb-4bit-use-double-quant", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--torchao-int8-weight-only", action="store_true")
    parser.add_argument("--use-deepspeed-zero2", action="store_true")
    parser.add_argument("--use-deepspeed-zero3", action="store_true")
    parser.add_argument("--zero3-init-flag", action="store_true")
    parser.add_argument("--zero3-save-16bit-model", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=100)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument(
        "--checkpoint-every-steps",
        type=int,
        default=50,
        help="Save a strict resumable checkpoint every N optimizer steps. Set 0 to disable periodic checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-at-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save a strict resumable checkpoint after each validation pass.",
    )
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=3,
        help="Keep this many recent strict checkpoints. Set <=0 to keep all.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Path to a strict checkpoint directory, or 'latest' under the experiment checkpoint root.",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="Resume from the latest strict checkpoint under the experiment directory when present.",
    )
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", type=str, default=DEFAULT_LORA_TARGET_MODULES)
    parser.add_argument(
        "--train-lora-module-filter",
        type=str,
        default="",
        help=(
            "Comma-separated LoRA module name fragments to keep trainable when "
            "continuing from --init-adapter-path. Empty/all keeps every LoRA module trainable."
        ),
    )
    parser.add_argument(
        "--train-projector",
        action="store_true",
        help="Also train selected non-LoRA audio/text projector parameters.",
    )
    parser.add_argument(
        "--projector-module-filter",
        type=str,
        default="audio_tower.output_proj,embed_audio",
        help=(
            "Comma-separated parameter name fragments to train/save when "
            "--train-projector is enabled."
        ),
    )
    return parser.parse_args()


def resolve_grpo_weight_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.ce_weight > 0:
        raise ValueError("GRPO v1 does not support CE-weighted training. Set --ce-weight 0.")
    key_weight = 0.0 if args.key_weight is None else args.key_weight
    if args.objective == "bleu_only":
        raw_weights = {"bleu": 1.0, "chrf": 0.0, "key": 0.0, "ce": 0.0}
    elif args.objective == "chrf_only":
        raw_weights = {"bleu": 0.0, "chrf": 1.0, "key": 0.0, "ce": 0.0}
    else:
        raw_weights = {
            "bleu": args.bleu_weight,
            "chrf": args.chrf_weight,
            "key": key_weight,
            "ce": 0.0,
        }
    for name, value in raw_weights.items():
        if value < 0:
            raise ValueError(f"{name} weight must be >= 0.")
    reward_total = raw_weights["bleu"] + raw_weights["chrf"] + raw_weights["key"]
    if reward_total <= 0:
        raise ValueError("At least one of BLEU/chrF/key weights must be > 0 for GRPO.")
    normalized = {
        "bleu": raw_weights["bleu"] / reward_total,
        "chrf": raw_weights["chrf"] / reward_total,
        "key": raw_weights["key"] / reward_total,
        "ce": 0.0,
    }
    return {
        "raw": raw_weights,
        "normalized": normalized,
        "reward_weights": {
            "bleu": normalized["bleu"],
            "chrf": normalized["chrf"],
            "key": normalized["key"],
        },
        "risk_objective_weight": 1.0,
        "ce_weight": 0.0,
        "key_match_mode": args.key_match_mode,
        "entity_reward_mode": args.entity_reward_mode,
        "entity_soft_tau": args.entity_soft_tau,
        "entity_embedding_model": args.entity_embedding_model,
        "entity_embedding_pooling": args.entity_embedding_pooling,
        "effective_gradient_checkpointing": should_enable_model_gradient_checkpointing(args),
        "clip_range": args.clip_range,
        "kl_coef": args.kl_coef,
    }


def format_float_component(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text.replace(".", "p")


def build_experiment_name(weight_config: dict[str, Any]) -> str:
    normalized = weight_config["normalized"]
    return (
        "grpo_"
        f"b{format_float_component(normalized['bleu'])}_"
        f"c{format_float_component(normalized['chrf'])}_"
        f"k{format_float_component(normalized['key'])}"
    )


def slice_batch(batch: dict[str, Any], start: int, end: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in batch.items():
        sliced[key] = value[start:end] if isinstance(value, torch.Tensor) else value
    return sliced


def prepare_scoring_inputs(
    processor: Any,
    sample: dict[str, Any],
    assistant_texts: list[str],
    accelerator: Any,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    prompt_messages = build_prompt_messages(sample, args.sampling_rate)
    prompt_inputs = processor.apply_chat_template(
        [prompt_messages],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        processor_kwargs={"sampling_rate": args.sampling_rate},
    )
    audio_token_id = get_audio_token_id(processor)
    pad_token_id = get_pad_token_id(processor)
    prompt_inputs = align_audio_feature_mask(prompt_inputs)
    prompt_inputs = trim_surplus_audio_tokens(prompt_inputs, audio_token_id, pad_token_id)
    prompt_length = prompt_inputs["input_ids"].size(-1)
    full_conversations = [
        prompt_messages
        + [{"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}]
        for assistant_text in assistant_texts
    ]
    full_inputs = processor.apply_chat_template(
        full_conversations,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
        processor_kwargs={"sampling_rate": args.sampling_rate},
    )
    max_scoring_length = prompt_length + max(int(args.max_new_tokens), 1)
    input_length = full_inputs["input_ids"].size(-1)
    if input_length > max_scoring_length:
        text_sequence_keys = {"input_ids", "attention_mask", "token_type_ids", "position_ids"}
        for key, value in list(full_inputs.items()):
            if (
                key in text_sequence_keys
                and
                isinstance(value, torch.Tensor)
                and value.ndim >= 2
                and value.size(-1) == input_length
            ):
                full_inputs[key] = value[..., :max_scoring_length]
    full_inputs = move_batch_to_device(
        full_inputs,
        accelerator.device,
        float_dtype=get_runtime_float_dtype(args),
        audio_token_id=audio_token_id,
        pad_token_id=pad_token_id,
    )
    return full_inputs, prompt_length


def score_prepared_teacher_forcing(
    model: torch.nn.Module,
    full_inputs: dict[str, Any],
    prompt_length: int,
    args: argparse.Namespace,
    require_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = full_inputs["input_ids"][:, 1:]
    seq_positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    max_target_position = max(prompt_length - 1, 0) + max(int(args.max_new_tokens), 1)
    target_mask = full_inputs["attention_mask"][:, 1:].bool() & (
        (seq_positions >= max(prompt_length - 1, 0))
        & (seq_positions < max_target_position)
    )

    context = contextlib.nullcontext() if require_grad else torch.inference_mode()
    with context:
        with gemma4_zero3_gathered_forward_context(model, args):
            outputs = model(**full_inputs)
        logits = outputs.logits[:, :-1, :]
        active_positions = target_mask.nonzero(as_tuple=False)
        if active_positions.numel() == 0:
            token_nll_sums = logits.new_zeros(labels.shape[0], dtype=torch.float32)
        else:
            active_rows = active_positions[:, 0]
            active_logits = logits[target_mask].float()
            active_labels = labels[target_mask]
            active_nll = F.cross_entropy(
                active_logits,
                active_labels,
                reduction="none",
            )
            token_nll_sums = logits.new_zeros(labels.shape[0], dtype=torch.float32)
            token_nll_sums.index_add_(0, active_rows, active_nll)
        token_counts = target_mask.sum(dim=1).clamp_min(1)
        avg_nll = token_nll_sums / token_counts
        avg_log_probs = -avg_nll
        avg_log_probs = torch.nan_to_num(avg_log_probs, nan=-1e4, posinf=0.0, neginf=-1e4)
        avg_nll = torch.nan_to_num(avg_nll, nan=1e4, posinf=1e4, neginf=1e4)
    return avg_log_probs, avg_nll


def score_assistant_texts_with_teacher_forcing_chunked(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    assistant_texts: list[str],
    accelerator: Any,
    args: argparse.Namespace,
    require_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not assistant_texts:
        empty = torch.empty(0, device=accelerator.device, dtype=torch.float32)
        return empty, empty
    full_inputs, prompt_length = prepare_scoring_inputs(
        processor=processor,
        sample=sample,
        assistant_texts=assistant_texts,
        accelerator=accelerator,
        args=args,
    )
    batch_size = max(args.score_batch_size, 1)
    avg_log_prob_chunks: list[torch.Tensor] = []
    avg_nll_chunks: list[torch.Tensor] = []
    for start in range(0, len(assistant_texts), batch_size):
        chunk_inputs = slice_batch(full_inputs, start, start + batch_size)
        avg_log_probs, avg_nll = score_prepared_teacher_forcing(
            model=model,
            full_inputs=chunk_inputs,
            prompt_length=prompt_length,
            args=args,
            require_grad=require_grad,
        )
        avg_log_prob_chunks.append(avg_log_probs)
        avg_nll_chunks.append(avg_nll)
    return torch.cat(avg_log_prob_chunks, dim=0), torch.cat(avg_nll_chunks, dim=0)


def set_active_adapter(
    model: torch.nn.Module,
    adapter_name: str,
    accelerator: Any | None = None,
) -> None:
    targets: list[Any] = [model]
    if accelerator is not None:
        try:
            targets.append(accelerator.unwrap_model(model))
        except Exception:
            pass
    module = getattr(model, "module", None)
    if module is not None:
        targets.append(module)

    seen: set[int] = set()
    applied = False
    for target in targets:
        if target is None or id(target) in seen:
            continue
        seen.add(id(target))
        if hasattr(target, "set_adapter"):
            target.set_adapter(adapter_name)
            applied = True
    if not applied:
        raise RuntimeError(f"Could not switch active adapter to '{adapter_name}'.")


@contextlib.contextmanager
def adapter_context(
    model: torch.nn.Module,
    adapter_name: str,
    restore_name: str,
    accelerator: Any | None = None,
):
    set_active_adapter(model, adapter_name, accelerator=accelerator)
    try:
        yield
    finally:
        set_active_adapter(model, restore_name, accelerator=accelerator)


def load_policy_model_with_reference(
    base_model: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, list[str], int, int]:
    model, target_modules, _, _ = apply_lora(base_model, args)
    if not hasattr(model, "load_adapter"):
        raise RuntimeError("Current PEFT version does not expose load_adapter for GRPO reference loading.")
    model.load_adapter(str(args.init_adapter_path), adapter_name="reference", is_trainable=False)
    set_active_adapter(model, "default")
    train_lora_module_filter = [
        item.strip()
        for item in str(args.train_lora_module_filter or "").split(",")
        if item.strip()
    ]
    if train_lora_module_filter and train_lora_module_filter != ["all"]:
        trainable_fragments = tuple(train_lora_module_filter)
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if ".lora_" not in name or ".default." not in name:
                continue
            if not any(fragment in name for fragment in trainable_fragments):
                parameter.requires_grad_(False)
    trainable_projector_parameter_names: list[str] = []
    if args.train_projector:
        projector_fragments = tuple(
            item.strip()
            for item in str(args.projector_module_filter or "").split(",")
            if item.strip()
        )
        if not projector_fragments:
            raise ValueError("--projector-module-filter must not be empty when --train-projector is enabled.")
        for name, parameter in model.named_parameters():
            if ".reference." in name:
                continue
            if any(fragment in name for fragment in projector_fragments):
                parameter.requires_grad_(True)
                trainable_projector_parameter_names.append(name)
        if not trainable_projector_parameter_names:
            raise ValueError(
                "No projector parameters matched --projector-module-filter: "
                f"{args.projector_module_filter}"
            )
    trainable_lora_parameter_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and ".lora_" in name and ".default." in name
    ]
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    model._grpo_trainable_lora_parameter_names = trainable_lora_parameter_names
    model._grpo_trainable_projector_parameter_names = trainable_projector_parameter_names
    return model, target_modules, trainable_params, total_params


def save_extra_trainable_parameters(
    model: torch.nn.Module,
    output_dir: Path,
    accelerator: Any,
) -> None:
    projector_names = list(getattr(model, "_grpo_trainable_projector_parameter_names", []))
    if not projector_names:
        return
    unwrapped = accelerator.unwrap_model(model)
    named_parameters = dict(unwrapped.named_parameters())
    payload = {
        name: named_parameters[name].detach().cpu()
        for name in projector_names
        if name in named_parameters
    }
    if payload:
        torch.save(payload, output_dir / "extra_trainable_projector.pt")


def save_policy_adapter(
    model: torch.nn.Module,
    output_dir: Path,
    accelerator: Any,
) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        staging_dir = output_dir.with_name(f"{output_dir.name}.tmp")
        if staging_dir.exists():
            import shutil

            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)
        peft_model = accelerator.unwrap_model(model)
        if hasattr(peft_model, "set_adapter"):
            peft_model.set_adapter("default")
        try:
            peft_model.save_pretrained(staging_dir, selected_adapters=["default"])
        except TypeError:
            peft_model.save_pretrained(staging_dir)
        save_extra_trainable_parameters(model, staging_dir, accelerator)
        if output_dir.exists():
            import shutil

            shutil.rmtree(output_dir, ignore_errors=True)
        staging_dir.replace(output_dir)
    accelerator.wait_for_everyone()


def checkpoint_root_for_experiment(experiment_dir: Path) -> Path:
    return experiment_dir / "checkpoints"


def latest_checkpoint_marker(checkpoint_root: Path) -> Path:
    return checkpoint_root / "latest_checkpoint.txt"


def resolve_resume_checkpoint(args: argparse.Namespace, experiment_dir: Path) -> Path | None:
    checkpoint_root = checkpoint_root_for_experiment(experiment_dir)
    requested = args.resume_from_checkpoint
    if requested is not None:
        if str(requested) == "latest":
            marker = latest_checkpoint_marker(checkpoint_root)
            if not marker.exists():
                raise ValueError(f"No latest checkpoint marker found at {marker}")
            return Path(marker.read_text(encoding="utf-8").strip())
        return requested
    if args.auto_resume:
        marker = latest_checkpoint_marker(checkpoint_root)
        if marker.exists():
            checkpoint_path = Path(marker.read_text(encoding="utf-8").strip())
            return checkpoint_path if checkpoint_path.exists() else None
    return None


def read_trainer_state(checkpoint_dir: Path) -> dict[str, Any]:
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        raise ValueError(f"Missing trainer state in checkpoint: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def write_latest_checkpoint_marker(checkpoint_root: Path, checkpoint_dir: Path) -> None:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    latest_checkpoint_marker(checkpoint_root).write_text(
        str(checkpoint_dir.resolve()) + "\n",
        encoding="utf-8",
    )


def prune_old_checkpoints(checkpoint_root: Path, keep_last: int) -> None:
    if keep_last <= 0 or not checkpoint_root.exists():
        return
    checkpoints = sorted(
        [
            path
            for path in checkpoint_root.glob("step_*")
            if path.is_dir() and (path / "trainer_state.json").exists()
        ],
        key=lambda path: int(path.name.split("_", 1)[1]) if path.name.split("_", 1)[1].isdigit() else -1,
    )
    for stale_path in checkpoints[:-keep_last]:
        shutil.rmtree(stale_path, ignore_errors=True)


def save_training_checkpoint(
    *,
    model: torch.nn.Module,
    accelerator: Any,
    checkpoint_root: Path,
    global_step: int,
    next_epoch: int,
    next_batch_index: int,
    best_score: float,
    skipped_nonfinite_steps: int,
    skipped_audio_errors: int,
    max_train_steps: int,
    args: argparse.Namespace,
    reason: str,
) -> Path:
    checkpoint_dir = checkpoint_root / f"step_{global_step:08d}"
    staging_dir = checkpoint_root / f".step_{global_step:08d}.tmp"
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    accelerator.save_state(str(staging_dir / "accelerator_state"))
    save_policy_adapter(model, staging_dir / "policy_adapter", accelerator)

    if accelerator.is_main_process:
        trainer_state = {
            "checkpoint_version": 1,
            "global_step": global_step,
            "next_epoch": next_epoch,
            "next_batch_index": next_batch_index,
            "best_score": best_score,
            "skipped_nonfinite_steps": skipped_nonfinite_steps,
            "skipped_audio_errors": skipped_audio_errors,
            "reference_adapter_path": str(args.init_adapter_path),
            "policy_adapter_name": "default",
            "reference_adapter_name": "reference",
            "reason": reason,
            "max_train_steps": max_train_steps,
            "num_train_epochs": args.num_train_epochs,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
        save_json(staging_dir / "trainer_state.json", trainer_state)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        staging_dir.replace(checkpoint_dir)
        write_latest_checkpoint_marker(checkpoint_root, checkpoint_dir)
        prune_old_checkpoints(checkpoint_root, args.keep_last_checkpoints)
    accelerator.wait_for_everyone()
    return checkpoint_dir


def checkpoint_next_position(
    epoch: int,
    batch_index: int,
    num_epoch_batches: int,
) -> tuple[int, int]:
    if batch_index >= num_epoch_batches:
        return epoch + 1, 1
    return epoch, batch_index + 1


def shuffled_epoch_samples(
    samples: list[dict[str, Any]],
    seed: int,
    epoch: int,
) -> list[dict[str, Any]]:
    shuffled = list(samples)
    epoch_rng = random.Random(int(seed) + int(epoch) * 1_000_003)
    epoch_rng.shuffle(shuffled)
    return shuffled


def is_recoverable_audio_error(exc: BaseException) -> bool:
    message = str(exc)
    recoverable_markers = (
        "Audio features and audio tokens do not match",
        "Audio file not found",
        "Failed to load audio",
        "input_features",
        "audio preprocessing",
    )
    return any(marker in message for marker in recoverable_markers)


def clear_accumulators(*accumulators: list[float]) -> None:
    for accumulator in accumulators:
        accumulator.clear()


def generate_candidates(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    accelerator: Any,
    args: argparse.Namespace,
) -> list[str]:
    from fca_grpo_risk_runtime import generate_candidate_texts

    set_active_adapter(model, "default", accelerator=accelerator)
    return generate_candidate_texts(model, processor, sample, accelerator, args)


def compute_grpo_loss_for_sample(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    accelerator: Any,
    args: argparse.Namespace,
    weight_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    set_active_adapter(model, "default", accelerator=accelerator)
    candidates = generate_candidates(model, processor, sample, accelerator, args)

    old_log_probs, _ = score_assistant_texts_with_teacher_forcing_chunked(
        model=model,
        processor=processor,
        sample=sample,
        assistant_texts=candidates,
        accelerator=accelerator,
        args=args,
        require_grad=False,
    )
    with adapter_context(model, "reference", "default", accelerator=accelerator):
        reference_log_probs, _ = score_assistant_texts_with_teacher_forcing_chunked(
            model=model,
            processor=processor,
            sample=sample,
            assistant_texts=candidates,
            accelerator=accelerator,
            args=args,
            require_grad=False,
        )
    current_log_probs, _ = score_assistant_texts_with_teacher_forcing_chunked(
        model=model,
        processor=processor,
        sample=sample,
        assistant_texts=candidates,
        accelerator=accelerator,
        args=args,
        require_grad=True,
    )

    rewards: list[float] = []
    bleu_scores: list[float] = []
    chrf_scores: list[float] = []
    key_scores: list[float] = []
    entity_scores: list[float] = []
    for candidate in candidates:
        reward, score_row = compute_candidate_reward(candidate, sample, weight_config)
        rewards.append(reward)
        bleu_scores.append(score_row["bleu"])
        chrf_scores.append(score_row["chrf"])
        key_scores.append(score_row["key_recall"])
        entity_scores.append(score_row.get("entity_score", score_row["key_recall"]))

    reward_tensor = torch.tensor(rewards, device=current_log_probs.device, dtype=torch.float32)
    if reward_tensor.numel() > 1:
        advantages = (reward_tensor - reward_tensor.mean()) / (reward_tensor.std(unbiased=False) + 1e-6)
    else:
        advantages = torch.zeros_like(reward_tensor)

    old_log_probs = old_log_probs.detach().float()
    reference_log_probs = reference_log_probs.detach().float()
    current_log_probs = current_log_probs.float()

    log_ratio = current_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    policy_loss = -surrogate.mean()

    ref_delta = reference_log_probs - current_log_probs
    kl_per_candidate = torch.exp(ref_delta) - ref_delta - 1.0
    kl_loss = kl_per_candidate.mean()

    total_loss = policy_loss + args.kl_coef * kl_loss
    total_loss = torch.nan_to_num(total_loss, nan=1.0, posinf=1.0, neginf=1.0)

    metrics = {
        "loss": float(total_loss.detach().cpu().item()),
        "policy_loss": float(policy_loss.detach().cpu().item()),
        "kl_loss": float(kl_loss.detach().cpu().item()),
        "mean_reward": float(reward_tensor.mean().detach().cpu().item()),
        "best_minus_mean_reward": float(
            reward_tensor.max().detach().cpu().item() - reward_tensor.mean().detach().cpu().item()
        ),
        "mean_advantage": float(advantages.mean().detach().cpu().item()),
        "advantage_std": float(advantages.std(unbiased=False).detach().cpu().item()),
        "mean_bleu": float(sum(bleu_scores) / len(bleu_scores) * 100.0),
        "mean_chrf": float(sum(chrf_scores) / len(chrf_scores) * 100.0),
        "mean_key_recall": float(sum(key_scores) / len(key_scores)),
        "mean_entity_score": float(sum(entity_scores) / len(entity_scores)),
        "policy_logprob": float(current_log_probs.mean().detach().cpu().item()),
        "reference_logprob": float(reference_log_probs.mean().detach().cpu().item()),
        "kl_mean": float(kl_per_candidate.mean().detach().cpu().item()),
    }
    return total_loss, metrics


def train() -> None:
    args = parse_args()
    if args.num_train_epochs <= 0:
        raise ValueError("--num-train-epochs must be >= 1.")
    if args.per_device_train_batch_size <= 0:
        raise ValueError("--per-device-train-batch-size must be >= 1.")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be >= 1.")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be >= 1.")
    if args.num_candidates < 2:
        raise ValueError("--num-candidates must be >= 2 for GRPO group-relative advantages.")
    if args.score_batch_size <= 0:
        raise ValueError("--score-batch-size must be >= 1.")
    if args.checkpoint_every_steps < 0:
        raise ValueError("--checkpoint-every-steps must be >= 0.")
    if args.use_deepspeed_zero2 and args.use_deepspeed_zero3:
        raise ValueError("--use-deepspeed-zero2 and --use-deepspeed-zero3 cannot both be enabled.")
    if args.use_deepspeed_zero3:
        raise ValueError("GRPO v1 does not support ZeRO-3. Use --use-deepspeed-zero2 instead.")
    if args.load_in_4bit and args.load_in_8bit:
        raise ValueError("--load-in-4bit and --load-in-8bit cannot both be enabled.")
    if args.ce_weight > 0:
        raise ValueError("GRPO v1 does not support CE-weighted training.")
    if not args.init_adapter_path.exists():
        raise ValueError(f"--init-adapter-path does not exist: {args.init_adapter_path}")

    weight_config = resolve_grpo_weight_config(args)
    weight_config["_entity_rewarder"] = build_entity_rewarder(args, weight_config)
    if weight_config["normalized"]["key"] > 0 and args.train_entity_path is None:
        raise ValueError("--train-entity-path is required when --key-weight > 0.")

    bind_local_cuda_device_from_env()
    patch_torch_finfo_for_quantized_gemma4(args)
    patch_torch_masked_scatter_for_quantized_gemma4(args)
    patch_torch_masked_fill_for_low_precision(args)
    seed_everything(args.seed)

    experiment_name = args.experiment_name or build_experiment_name(weight_config)
    experiment_dir = args.output_root / experiment_name
    adapter_best_dir = experiment_dir / "adapter_best"
    adapter_last_dir = experiment_dir / "adapter_last"
    checkpoint_root = checkpoint_root_for_experiment(experiment_dir)
    train_metrics_path = experiment_dir / "train_metrics.jsonl"
    val_metrics_path = experiment_dir / "val_metrics.jsonl"
    run_config_path = experiment_dir / "run_config.json"
    tensorboard_dir = args.tensorboard_dir or (experiment_dir / "tensorboard")
    resume_checkpoint_dir = resolve_resume_checkpoint(args, experiment_dir)
    resume_trainer_state: dict[str, Any] | None = None
    if resume_checkpoint_dir is not None:
        if not resume_checkpoint_dir.exists():
            raise ValueError(f"--resume-from-checkpoint does not exist: {resume_checkpoint_dir}")
        resume_trainer_state = read_trainer_state(resume_checkpoint_dir)
        saved_reference_path = str(resume_trainer_state.get("reference_adapter_path", ""))
        if saved_reference_path and saved_reference_path != str(args.init_adapter_path):
            raise ValueError(
                "Resume checkpoint was created with a different reference adapter. "
                f"checkpoint={saved_reference_path}, current={args.init_adapter_path}"
            )

    train_samples = attach_sidecar_annotations(
        load_samples(args.train_data_path, args, args.limit_train_samples),
        load_key_sidecar(args.train_entity_path),
        load_entity_sidecar(args.train_entity_path),
    )
    val_samples = attach_sidecar_annotations(
        load_samples(args.val_data_path, args, args.limit_val_samples),
        load_key_sidecar(args.val_entity_path),
        load_entity_sidecar(args.val_entity_path),
    )
    val_key_labels_available = args.val_entity_path is not None

    accelerator = build_accelerator(args)
    if args.use_deepspeed_zero2 or args.use_deepspeed_zero3:
        plugin = accelerator.state.deepspeed_plugin
        plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        plugin.deepspeed_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps

    if accelerator.is_main_process:
        experiment_dir.mkdir(parents=True, exist_ok=True)
        if resume_checkpoint_dir is None:
            for metrics_path in (train_metrics_path, val_metrics_path):
                if metrics_path.exists():
                    metrics_path.unlink()
        if resume_checkpoint_dir is not None and run_config_path.exists():
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        else:
            run_config = {
                "experiment_name": experiment_name,
                "base_model_path": args.base_model_path,
                "init_adapter_path": str(args.init_adapter_path),
                "reference_adapter_path": str(args.init_adapter_path),
                "train_data_path": str(args.train_data_path),
                "val_data_path": str(args.val_data_path),
                "train_entity_path": str(args.train_entity_path) if args.train_entity_path else None,
                "val_entity_path": str(args.val_entity_path) if args.val_entity_path else None,
                "objective": args.objective,
                "reward_weights": weight_config["raw"],
                "normalized_reward_weights": weight_config["normalized"],
                "group_advantage_normalization": "zscore_per_prompt_group",
                "clip_range": args.clip_range,
                "kl_coef": args.kl_coef,
                "score_batch_size": args.score_batch_size,
                "key_match_mode": args.key_match_mode,
                "entity_reward_mode": args.entity_reward_mode,
                "entity_soft_tau": args.entity_soft_tau,
                "entity_embedding_model": args.entity_embedding_model,
                "entity_embedding_pooling": args.entity_embedding_pooling,
                "ner_tokenizer_model": args.ner_tokenizer_model,
                "ner_model": args.ner_model,
                "output_dir": str(experiment_dir),
                "best_checkpoint_path": str(adapter_best_dir),
                "last_checkpoint_path": str(adapter_last_dir),
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "val_max_new_tokens": args.val_max_new_tokens,
                "num_candidates": args.num_candidates,
                "mixed_precision": args.mixed_precision,
                "torch_dtype": args.torch_dtype,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "lr_scheduler_type": args.lr_scheduler_type,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "eval_batch_size": args.eval_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "gradient_checkpointing": args.gradient_checkpointing,
                "effective_gradient_checkpointing": weight_config["effective_gradient_checkpointing"],
                "use_deepspeed_zero2": args.use_deepspeed_zero2,
                "use_deepspeed_zero3": args.use_deepspeed_zero3,
                "zero3_init_flag": args.zero3_init_flag,
                "zero3_save_16bit_model": args.zero3_save_16bit_model,
                "load_in_4bit": args.load_in_4bit,
                "load_in_8bit": args.load_in_8bit,
                "torchao_int8_weight_only": args.torchao_int8_weight_only,
                "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
                "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
                "bnb_4bit_use_double_quant": args.bnb_4bit_use_double_quant,
                "train_projector": args.train_projector,
                "projector_module_filter": args.projector_module_filter,
                "lora": {
                    "enabled": True,
                    "r": args.lora_r,
                    "alpha": args.lora_alpha,
                    "dropout": args.lora_dropout,
                    "requested_target_modules": args.lora_target_modules,
                    "train_module_filter": args.train_lora_module_filter,
                    "init_adapter_path": str(args.init_adapter_path),
                    "reference_adapter_name": "reference",
                    "policy_adapter_name": "default",
                },
                "quantization": {},
            }
        run_config["resume"] = {
            "enabled": resume_checkpoint_dir is not None,
            "checkpoint_path": str(resume_checkpoint_dir) if resume_checkpoint_dir is not None else None,
            "auto_resume": args.auto_resume,
        }
        save_json(run_config_path, run_config)

    tb_writer = None
    if args.enable_tensorboard and accelerator.is_main_process:
        SummaryWriter = get_summary_writer()
        if SummaryWriter is None:
            raise RuntimeError("TensorBoard support requires tensorboard to be installed.")
        tb_writer = SummaryWriter(log_dir=str(tensorboard_dir))

    processor, base_model = load_model_and_processor(args, accelerator=accelerator)
    model, target_modules, trainable_params, total_params = load_policy_model_with_reference(
        base_model,
        args,
    )
    quantization_summary = collect_quantization_summary(model)
    if accelerator.is_main_process:
        run_config.update(
            {
                "target_modules": target_modules,
                "trainable_params": trainable_params,
                "total_params": total_params,
                "trainable_param_ratio": (
                    trainable_params / total_params if total_params else None
                ),
                "quantization": quantization_summary,
                "trainable_lora_parameter_count": len(
                    getattr(model, "_grpo_trainable_lora_parameter_names", [])
                ),
                "trainable_lora_parameter_name_sample": getattr(
                    model, "_grpo_trainable_lora_parameter_names", []
                )[:20],
                "trainable_projector_parameter_count": len(
                    getattr(model, "_grpo_trainable_projector_parameter_names", [])
                ),
                "trainable_projector_parameter_names": getattr(
                    model, "_grpo_trainable_projector_parameter_names", []
                ),
            }
        )
        run_config["lora"]["resolved_target_modules"] = target_modules
        save_json(run_config_path, run_config)
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_batches_per_epoch = math.ceil(
        len(train_samples) / max(args.per_device_train_batch_size, 1)
    )
    update_steps_per_epoch = math.ceil(
        train_batches_per_epoch / max(args.gradient_accumulation_steps, 1)
    )
    max_train_steps = args.max_train_steps or (args.num_train_epochs * update_steps_per_epoch)
    warmup_steps = int(max_train_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_train_steps,
    )

    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    if args.use_deepspeed_zero3:
        patch_gemma4_zero3_forward(accelerator.unwrap_model(model), args)
        patch_gemma4_zero3_forward(model, args)
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]

    global_step = 0
    best_score = float("-inf")
    start_epoch = 1
    start_batch_index = 1
    skipped_nonfinite_steps = 0
    skipped_audio_errors = 0
    if resume_checkpoint_dir is not None:
        accelerator.load_state(str(resume_checkpoint_dir / "accelerator_state"))
        set_active_adapter(model, "default", accelerator=accelerator)
        if resume_trainer_state is None:
            raise RuntimeError("Internal error: resume checkpoint state was not loaded.")
        global_step = int(resume_trainer_state.get("global_step", 0))
        best_score = float(resume_trainer_state.get("best_score", float("-inf")))
        start_epoch = int(resume_trainer_state.get("next_epoch", 1))
        start_batch_index = int(resume_trainer_state.get("next_batch_index", 1))
        skipped_nonfinite_steps = int(resume_trainer_state.get("skipped_nonfinite_steps", 0))
        skipped_audio_errors = int(resume_trainer_state.get("skipped_audio_errors", 0))

    if accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "experiment_name": experiment_name,
                    "target_modules": target_modules,
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "max_train_steps": max_train_steps,
                    "normalized_reward_weights": weight_config["normalized"],
                    "effective_gradient_checkpointing": weight_config["effective_gradient_checkpointing"],
                    "quantization": quantization_summary,
                    "policy_adapter": "default",
                    "reference_adapter": "reference",
                    "resume_checkpoint": str(resume_checkpoint_dir) if resume_checkpoint_dir is not None else None,
                    "start_epoch": start_epoch,
                    "start_batch_index": start_batch_index,
                    "global_step": global_step,
                    "trainable_lora_parameter_count": len(
                        getattr(model, "_grpo_trainable_lora_parameter_names", [])
                    ),
                    "trainable_lora_parameter_name_sample": getattr(
                        model, "_grpo_trainable_lora_parameter_names", []
                    )[:10],
                    "trainable_projector_parameter_count": len(
                        getattr(model, "_grpo_trainable_projector_parameter_names", [])
                    ),
                    "trainable_projector_parameter_names": getattr(
                        model, "_grpo_trainable_projector_parameter_names", []
                    ),
                },
                ensure_ascii=False,
            )
        )

    optimizer.zero_grad()
    train_progress_bar = tqdm(
        total=max_train_steps,
        desc="Training",
        initial=min(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )

    accumulated_losses: list[float] = []
    accumulated_policy_losses: list[float] = []
    accumulated_kl_losses: list[float] = []
    accumulated_rewards: list[float] = []
    accumulated_best_minus_mean_rewards: list[float] = []
    accumulated_advantages: list[float] = []
    accumulated_advantage_stds: list[float] = []
    accumulated_bleu: list[float] = []
    accumulated_chrf: list[float] = []
    accumulated_key_recall: list[float] = []
    accumulated_entity_score: list[float] = []
    accumulated_policy_logprob: list[float] = []
    accumulated_reference_logprob: list[float] = []
    accumulated_kl_mean: list[float] = []
    stop_training = False
    last_epoch = max(1, min(start_epoch, args.num_train_epochs))
    for epoch in range(start_epoch, args.num_train_epochs + 1):
        last_epoch = epoch
        shuffled_samples = shuffled_epoch_samples(train_samples, args.seed, epoch)
        epoch_batches = list(iter_batches(shuffled_samples, args.per_device_train_batch_size))

        for batch_index, batch in enumerate(epoch_batches, start=1):
            if epoch == start_epoch and batch_index < start_batch_index:
                continue
            set_active_adapter(model, "default", accelerator=accelerator)
            model.train()
            sample_losses: list[torch.Tensor] = []
            batch_metrics: list[dict[str, float]] = []

            for sample in batch:
                try:
                    sample_loss, sample_metrics = compute_grpo_loss_for_sample(
                        model=model,
                        processor=processor,
                        sample=sample,
                        accelerator=accelerator,
                        args=args,
                        weight_config=weight_config,
                    )
                except (FileNotFoundError, RuntimeError, ValueError) as exc:
                    if not is_recoverable_audio_error(exc):
                        raise
                    skipped_audio_errors += 1
                    if accelerator.is_main_process:
                        print(
                            json.dumps(
                                {
                                    "event": "skip_recoverable_audio_sample",
                                    "epoch": epoch,
                                    "batch_index": batch_index,
                                    "global_step": global_step,
                                    "sample_id": sample.get("id", sample.get("key", "<unknown>")),
                                    "audio_path": sample.get("audio_path", "<unknown>"),
                                    "error": str(exc),
                                },
                                ensure_ascii=False,
                            )
                        )
                    continue
                sample_losses.append(sample_loss)
                batch_metrics.append(sample_metrics)

            if not sample_losses:
                optimizer.zero_grad()
                continue

            batch_loss = torch.stack(sample_losses).mean()
            if not torch.isfinite(batch_loss):
                skipped_nonfinite_steps += 1
                optimizer.zero_grad()
                if accelerator.is_main_process:
                    print(
                        json.dumps(
                            {
                                "event": "skip_nonfinite_batch",
                                "epoch": epoch,
                                "batch_index": batch_index,
                                "global_step": global_step,
                            },
                            ensure_ascii=False,
                        )
                    )
                continue

            accelerator.backward(batch_loss / args.gradient_accumulation_steps)

            accumulated_losses.append(float(batch_loss.detach().cpu().item()))
            accumulated_policy_losses.extend(metric["policy_loss"] for metric in batch_metrics)
            accumulated_kl_losses.extend(metric["kl_loss"] for metric in batch_metrics)
            accumulated_rewards.extend(metric["mean_reward"] for metric in batch_metrics)
            accumulated_best_minus_mean_rewards.extend(
                metric["best_minus_mean_reward"] for metric in batch_metrics
            )
            accumulated_advantages.extend(metric["mean_advantage"] for metric in batch_metrics)
            accumulated_advantage_stds.extend(metric["advantage_std"] for metric in batch_metrics)
            accumulated_bleu.extend(metric["mean_bleu"] for metric in batch_metrics)
            accumulated_chrf.extend(metric["mean_chrf"] for metric in batch_metrics)
            accumulated_key_recall.extend(metric["mean_key_recall"] for metric in batch_metrics)
            accumulated_entity_score.extend(
                metric.get("mean_entity_score", metric["mean_key_recall"]) for metric in batch_metrics
            )
            accumulated_policy_logprob.extend(metric["policy_logprob"] for metric in batch_metrics)
            accumulated_reference_logprob.extend(
                metric["reference_logprob"] for metric in batch_metrics
            )
            accumulated_kl_mean.extend(metric["kl_mean"] for metric in batch_metrics)

            should_step = (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == len(epoch_batches)
            )
            if not should_step:
                continue

            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(trainable_parameters, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            train_progress_bar.update(1)

            train_row = {
                "step": global_step,
                "epoch": epoch,
                "loss": sum(accumulated_losses) / len(accumulated_losses),
                "policy_loss": sum(accumulated_policy_losses) / len(accumulated_policy_losses),
                "kl_loss": sum(accumulated_kl_losses) / len(accumulated_kl_losses),
                "mean_reward": sum(accumulated_rewards) / len(accumulated_rewards),
                "best_minus_mean_reward": sum(accumulated_best_minus_mean_rewards)
                / len(accumulated_best_minus_mean_rewards),
                "mean_advantage": sum(accumulated_advantages) / len(accumulated_advantages),
                "advantage_std": sum(accumulated_advantage_stds) / len(accumulated_advantage_stds),
                "mean_bleu": sum(accumulated_bleu) / len(accumulated_bleu),
                "mean_chrf": sum(accumulated_chrf) / len(accumulated_chrf),
                "mean_key_recall": sum(accumulated_key_recall) / len(accumulated_key_recall),
                "mean_entity_score": sum(accumulated_entity_score) / len(accumulated_entity_score),
                "policy_logprob": sum(accumulated_policy_logprob) / len(accumulated_policy_logprob),
                "reference_logprob": sum(accumulated_reference_logprob)
                / len(accumulated_reference_logprob),
                "kl_mean": sum(accumulated_kl_mean) / len(accumulated_kl_mean),
                "learning_rate": float(lr_scheduler.get_last_lr()[0]),
                "bleu_weight": weight_config["normalized"]["bleu"],
                "chrf_weight": weight_config["normalized"]["chrf"],
                "key_weight": weight_config["normalized"]["key"],
                "kl_coef": args.kl_coef,
                "clip_range": args.clip_range,
                "checkpoint_path": str(adapter_best_dir),
                "skipped_nonfinite_steps": skipped_nonfinite_steps,
                "skipped_audio_errors": skipped_audio_errors,
            }
            next_epoch, next_batch_index = checkpoint_next_position(
                epoch,
                batch_index,
                len(epoch_batches),
            )
            if accelerator.is_main_process:
                append_jsonl(train_metrics_path, train_row)
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", train_row["loss"], global_step)
                    tb_writer.add_scalar("train/policy_loss", train_row["policy_loss"], global_step)
                    tb_writer.add_scalar("train/kl_loss", train_row["kl_loss"], global_step)
                    tb_writer.add_scalar(
                        "train/mean_reward_raw", train_row["mean_reward"], global_step
                    )
                    tb_writer.add_scalar(
                        "train/best_minus_mean_reward",
                        train_row["best_minus_mean_reward"],
                        global_step,
                    )
                    tb_writer.add_scalar(
                        "train/mean_advantage", train_row["mean_advantage"], global_step
                    )
                    tb_writer.add_scalar(
                        "train/advantage_std", train_row["advantage_std"], global_step
                    )
                    tb_writer.add_scalar("train/mean_bleu", train_row["mean_bleu"], global_step)
                    tb_writer.add_scalar("train/mean_chrf", train_row["mean_chrf"], global_step)
                    tb_writer.add_scalar(
                        "train/mean_key_recall", train_row["mean_key_recall"], global_step
                    )
                    tb_writer.add_scalar(
                        "train/mean_entity_score", train_row["mean_entity_score"], global_step
                    )
                    if not args.disable_kl_monitor:
                        tb_writer.add_scalar(
                            "train/policy_logprob", train_row["policy_logprob"], global_step
                        )
                        tb_writer.add_scalar(
                            "train/reference_logprob",
                            train_row["reference_logprob"],
                            global_step,
                        )
                        tb_writer.add_scalar("train/kl_mean", train_row["kl_mean"], global_step)
                        tb_writer.add_scalar("train/kl_coef", train_row["kl_coef"], global_step)
                if args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
                    print(json.dumps(train_row, ensure_ascii=False))

            clear_accumulators(
                accumulated_losses,
                accumulated_policy_losses,
                accumulated_kl_losses,
                accumulated_rewards,
                accumulated_best_minus_mean_rewards,
                accumulated_advantages,
                accumulated_advantage_stds,
                accumulated_bleu,
                accumulated_chrf,
                accumulated_key_recall,
                accumulated_entity_score,
                accumulated_policy_logprob,
                accumulated_reference_logprob,
                accumulated_kl_mean,
            )

            should_checkpoint = (
                args.checkpoint_every_steps > 0
                and global_step > 0
                and global_step % args.checkpoint_every_steps == 0
            )
            if should_checkpoint:
                save_training_checkpoint(
                    model=model,
                    accelerator=accelerator,
                    checkpoint_root=checkpoint_root,
                    global_step=global_step,
                    next_epoch=next_epoch,
                    next_batch_index=next_batch_index,
                    best_score=best_score,
                    skipped_nonfinite_steps=skipped_nonfinite_steps,
                    skipped_audio_errors=skipped_audio_errors,
                    max_train_steps=max_train_steps,
                    args=args,
                    reason="periodic",
                )

            should_eval = args.eval_every_steps > 0 and global_step % args.eval_every_steps == 0
            if should_eval:
                set_active_adapter(model, "default", accelerator=accelerator)
                val_row = evaluate_validation(
                    model=model,
                    processor=processor,
                    samples=val_samples,
                    accelerator=accelerator,
                    args=args,
                    global_step=global_step,
                    epoch=epoch,
                    weight_config=weight_config,
                    vocab_size=get_vocab_size(processor),
                    key_labels_available=val_key_labels_available,
                    checkpoint_path=str(adapter_best_dir),
                )
                if accelerator.is_main_process:
                    append_jsonl(val_metrics_path, val_row)
                    if tb_writer is not None:
                        tb_writer.add_scalar("val/eval_bleu", val_row["eval_bleu"], global_step)
                        tb_writer.add_scalar("val/eval_chrf", val_row["eval_chrf"], global_step)
                        tb_writer.add_scalar(
                            "val/eval_key_recall", val_row["eval_key_recall"], global_step
                        )
                        tb_writer.add_scalar(
                            "val/eval_entity_score", val_row["eval_entity_score"], global_step
                        )
                        tb_writer.add_scalar(
                            "val/eval_len_ratio", val_row["eval_len_ratio"], global_step
                        )
                        tb_writer.add_scalar(
                            "val/selection_score", val_row["selection_score"], global_step
                        )
                    print(json.dumps(val_row, ensure_ascii=False))
                if val_row["selection_score"] > best_score:
                    best_score = val_row["selection_score"]
                    save_policy_adapter(model, adapter_best_dir, accelerator)
                if args.checkpoint_at_eval:
                    save_training_checkpoint(
                        model=model,
                        accelerator=accelerator,
                        checkpoint_root=checkpoint_root,
                        global_step=global_step,
                        next_epoch=next_epoch,
                        next_batch_index=next_batch_index,
                        best_score=best_score,
                        skipped_nonfinite_steps=skipped_nonfinite_steps,
                        skipped_audio_errors=skipped_audio_errors,
                        max_train_steps=max_train_steps,
                        args=args,
                        reason="eval",
                    )

            if global_step >= max_train_steps:
                stop_training = True
                break

        if stop_training:
            break

    set_active_adapter(model, "default", accelerator=accelerator)
    final_val_row = evaluate_validation(
        model=model,
        processor=processor,
        samples=val_samples,
        accelerator=accelerator,
        args=args,
        global_step=global_step,
        epoch=last_epoch,
        weight_config=weight_config,
        vocab_size=get_vocab_size(processor),
        key_labels_available=val_key_labels_available,
        checkpoint_path=str(adapter_best_dir),
    )
    if accelerator.is_main_process:
        append_jsonl(val_metrics_path, final_val_row)
        if tb_writer is not None:
            tb_writer.add_scalar("val/eval_bleu", final_val_row["eval_bleu"], global_step)
            tb_writer.add_scalar("val/eval_chrf", final_val_row["eval_chrf"], global_step)
            tb_writer.add_scalar(
                "val/eval_key_recall", final_val_row["eval_key_recall"], global_step
            )
            tb_writer.add_scalar(
                "val/eval_entity_score", final_val_row["eval_entity_score"], global_step
            )
            tb_writer.add_scalar(
                "val/eval_len_ratio", final_val_row["eval_len_ratio"], global_step
            )
            tb_writer.add_scalar(
                "val/selection_score", final_val_row["selection_score"], global_step
            )
        print(json.dumps(final_val_row, ensure_ascii=False))
    if final_val_row["selection_score"] > best_score:
        best_score = final_val_row["selection_score"]
        save_policy_adapter(model, adapter_best_dir, accelerator)

    save_policy_adapter(model, adapter_last_dir, accelerator)
    save_training_checkpoint(
        model=model,
        accelerator=accelerator,
        checkpoint_root=checkpoint_root,
        global_step=global_step,
        next_epoch=args.num_train_epochs + 1,
        next_batch_index=1,
        best_score=best_score,
        skipped_nonfinite_steps=skipped_nonfinite_steps,
        skipped_audio_errors=skipped_audio_errors,
        max_train_steps=max_train_steps,
        args=args,
        reason="final",
    )
    train_progress_bar.close()
    if tb_writer is not None:
        tb_writer.close()

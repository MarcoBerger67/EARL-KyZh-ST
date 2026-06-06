from __future__ import annotations

import argparse
import math
import inspect
import json
import logging
import os
import random
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.pytorch_utils import Conv1D
from transformers.trainer_utils import get_last_checkpoint


DEFAULT_BASE_MODEL_PATH = "gemma-4-E2B-it"
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)
PLACEHOLDER_TRANSLATION = "<Simplified Chinese translation>"
DEFAULT_PROMPT = (
    "You are a strict Kyrgyz-to-Chinese speech translation system.\n\n"
    "Translate the input speech segment into fluent Simplified Chinese.\n\n"
    "Rules:\n"
    "1. Output only the final translation.\n"
    "2. Output one single line only.\n"
    "3. Output the complete translation of the entire speech segment.\n"
    "4. Do not omit any translated content.\n"
    "5. Do not stop after translating only the beginning of the sentence.\n"
    "6. Output Simplified Chinese only. Do not output Kyrgyz, English, "
    "explanations, notes, labels, speaker tags, or markdown.\n"
    '7. Do not output prefixes or suffixes such as "Key words:", "Line:", '
    '"Translation:", "Answer:", or quotation marks around the answer.\n'
    "8. Preserve the meaning faithfully. Do not summarize, rewrite, expand, "
    "or invent content not supported by the speech.\n"
    "9. Preserve names, places, organizations, and numbers as accurately as possible.\n"
    "10. Write numbers using Arabic numerals, e.g. 1.7, 3, 80%.\n"
    "11. If some part is unclear, translate conservatively based on the audio and "
    "context, and do not hallucinate."
)

warnings.filterwarnings(
    "ignore",
    message=r"Kwargs passed to `processor\.__call__` have to be in `processor_kwargs` dict, not in `\*\*kwargs`",
)


PROCESSOR_KWARGS_NOISE = (
    "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict"
)


class _SuppressProcessorKwargsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return PROCESSOR_KWARGS_NOISE not in record.getMessage()


def suppress_processor_kwargs_noise() -> None:
    log_filter = _SuppressProcessorKwargsFilter()
    logger_names = ("transformers", "transformers.processing_utils")
    for logger_name in logger_names:
        logger = logging.getLogger(logger_name)
        logger.addFilter(log_filter)
        for handler in logger.handlers:
            handler.addFilter(log_filter)
    root_logger = logging.getLogger()
    root_logger.addFilter(log_filter)
    for handler in root_logger.handlers:
        handler.addFilter(log_filter)


suppress_processor_kwargs_noise()


def get_sacrebleu():
    try:
        import sacrebleu
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sacrebleu is required for generation metrics. Install it with `pip install sacrebleu`."
        ) from exc
    return sacrebleu


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train gemma-4-E2B-it with bf16 LoRA supervised fine-tuning."
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=DEFAULT_BASE_MODEL_PATH,
        help="Local or remote Hugging Face base model directory.",
    )
    parser.add_argument(
        "--train-data-path",
        type=Path,
        default=(
            root_dir
            / "data"
            / "converted_testt_format"
            / "train_ky2zh_full285h_stage3.cleaned.jsonl"
        ),
        help="Training JSONL in Gemma chat/audio format.",
    )
    parser.add_argument(
        "--val-data-path",
        type=Path,
        default=root_dir / "data" / "converted_testt_format" / "val_ky2zh_full285h.jsonl",
        help="Validation JSONL in Gemma chat/audio format.",
    )
    parser.add_argument(
        "--test-data-path",
        type=Path,
        default=root_dir / "data" / "converted_testt_format" / "testt.jsonl",
        help="Test JSONL path kept in run_config for downstream evaluation commands.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root_dir / "model" / "sft",
        help="Root directory for SFT experiment outputs.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="sft_gemma4_e2b_lora_stage3",
        help="Experiment directory name under output-root.",
    )
    parser.add_argument(
        "--audio-prefix-from",
        type=str,
        default="",
        help="Prefix to rewrite from the source audio path. Empty string disables rewriting.",
    )
    parser.add_argument(
        "--audio-prefix-to",
        type=str,
        default="",
        help="Prefix to rewrite to the target audio path.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Training micro-batch size per device.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        help="Evaluation micro-batch size per device.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of micro-batches per optimizer update.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=-1,
        help="Optional cap on optimizer update steps. Use -1 to disable.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for bf16 LoRA SFT.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="Weight decay.",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Fraction of update steps used for warmup.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        default="cosine",
        help="Learning-rate scheduler type.",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Sampling rate used when loading audio via the processor chat template.",
    )
    parser.add_argument(
        "--metric-max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens used for train/val BLEU and chrF evaluation.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum tokenized length passed to the processor.",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=None,
        help="Optional separate max length for prompt-only tokenization. Defaults to max-seq-length.",
    )
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["no", "fp16", "bf16"],
        default="bf16",
        help="Mixed precision mode for Trainer.",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help='Device map for from_pretrained. Use "none" to disable.',
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help='Optional attention implementation, for example "flash_attention_2".',
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--enable-tensorboard",
        action="store_true",
        help="Enable TensorBoard logging.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="Optional override for the TensorBoard log directory.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=200,
        help="Checkpoint save interval in optimizer steps. Use <= 0 to save every epoch.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=200,
        help="Evaluation interval in optimizer steps. Use <= 0 to evaluate every epoch.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Logging interval in optimizer steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of Trainer checkpoints to keep.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Optional Trainer checkpoint path to resume from.",
    )
    parser.add_argument(
        "--limit-train-samples",
        type=int,
        default=None,
        help="Optional limit for training samples.",
    )
    parser.add_argument(
        "--limit-val-samples",
        type=int,
        default=None,
        help="Optional limit for validation samples.",
    )
    parser.add_argument(
        "--metric-train-limit",
        type=int,
        default=256,
        help="Optional cap on the number of train samples used when computing BLEU/chrF.",
    )
    parser.add_argument(
        "--metric-val-limit",
        type=int,
        default=256,
        help="Optional cap on the number of val samples used when computing BLEU/chrF.",
    )
    parser.add_argument(
        "--optim",
        choices=["paged_adamw_8bit", "adamw_torch"],
        default="paged_adamw_8bit",
        help="Optimizer used by Trainer.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.05,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default=DEFAULT_LORA_TARGET_MODULES,
        help="Comma-separated LoRA target module names.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Optional compatibility path: load the base model with bitsandbytes 4-bit quantization.",
    )
    parser.add_argument(
        "--bnb-4bit-quant-type",
        choices=["nf4", "fp4"],
        default="nf4",
        help="4-bit quantization type for bitsandbytes.",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
        help="Compute dtype used inside the 4-bit layers.",
    )
    parser.add_argument(
        "--bnb-4bit-use-double-quant",
        action="store_true",
        help="Enable nested quantization in bitsandbytes.",
    )
    return parser.parse_args()


def resolve_torch_dtype(dtype_name: str) -> str | torch.dtype:
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def resolve_explicit_torch_dtype(dtype_name: str) -> torch.dtype:
    return getattr(torch, dtype_name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def bind_local_cuda_device_from_env() -> None:
    if not torch.cuda.is_available():
        return
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)


def normalize_text(text: str) -> str:
    text = text.strip()
    if text.startswith("Translation:"):
        text = text[len("Translation:") :].strip()
    return " ".join(text.split())


def rewrite_audio_path(audio_path: str, prefix_from: str, prefix_to: str) -> str:
    if not prefix_from:
        return audio_path
    if audio_path.startswith(prefix_to):
        return audio_path
    if not audio_path.startswith(prefix_from):
        raise ValueError(
            f"Audio path '{audio_path}' does not start with expected prefix '{prefix_from}'."
        )
    return prefix_to + audio_path.removeprefix(prefix_from)


def extract_reference(gt: str) -> str:
    if "Translation:" not in gt:
        raise ValueError("Missing 'Translation:' marker in ground truth.")
    text = gt.rsplit("Translation:", 1)[1].strip()
    if not text:
        raise ValueError("Empty translation extracted from ground truth.")
    return text


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        merged = " ".join(texts).strip()
        if merged:
            return merged
    raise ValueError("Could not extract text content from message.")


def extract_audio_path_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "audio":
                continue
            for key in ("path", "audio", "url"):
                value = item.get(key)
                if value:
                    return value
    raise ValueError("Could not find audio content in messages.")


def extract_prompt_from_messages(messages: list[dict[str, Any]]) -> str:
    # Ignore any prompt text embedded in the dataset and force a unified instruction
    # for SFT data preparation.
    return DEFAULT_PROMPT


def parse_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, str] | None:
    if {"key", "audio", "gt"}.issubset(record):
        reference = extract_reference(record["gt"])
        if reference == PLACEHOLDER_TRANSLATION:
            return None
        return {
            "id": record["key"],
            "audio_path": rewrite_audio_path(
                record["audio"], args.audio_prefix_from, args.audio_prefix_to
            ),
            "prompt": DEFAULT_PROMPT,
            "reference": normalize_text(reference),
        }

    if {"id", "messages"}.issubset(record):
        messages = record["messages"]
        if not isinstance(messages, list) or len(messages) < 2:
            raise ValueError("Messages format requires at least user and assistant turns.")
        reference = extract_text_from_content(messages[-1].get("content"))
        if reference == PLACEHOLDER_TRANSLATION:
            return None
        return {
            "id": record["id"],
            "audio_path": rewrite_audio_path(
                extract_audio_path_from_messages(messages),
                args.audio_prefix_from,
                args.audio_prefix_to,
            ),
            "prompt": extract_prompt_from_messages(messages),
            "reference": normalize_text(reference),
        }

    raise ValueError(
        "Unsupported input record format. Expected either key/audio/gt or id/messages."
    )


def load_samples(
    data_path: Path, args: argparse.Namespace, limit: int | None
) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            sample = parse_record(record, args)
            if sample is None:
                continue
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"No valid samples were loaded from {data_path}.")
    return samples


def build_prompt_messages(sample: dict[str, str]) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "path": sample["audio_path"]},
                {"type": "text", "text": sample["prompt"]},
            ],
        }
    ]


def build_training_messages(sample: dict[str, str]) -> list[dict[str, Any]]:
    return build_prompt_messages(sample) + [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": sample["reference"]}],
        }
    ]


def get_requested_target_modules(args: argparse.Namespace) -> list[str]:
    modules = [item.strip() for item in args.lora_target_modules.split(",")]
    modules = [item for item in modules if item]
    if not modules:
        raise ValueError("At least one LoRA target module must be specified.")
    return modules


def resolve_lora_target_modules(model: torch.nn.Module, requested: list[str]) -> list[str]:
    supported_types = (
        torch.nn.Linear,
        torch.nn.Embedding,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.MultiheadAttention,
        Conv1D,
    )
    named_modules = dict(model.named_modules())
    resolved: list[str] = []

    for requested_name in requested:
        matched_any = False
        for module_name, module in named_modules.items():
            if not (module_name == requested_name or module_name.endswith(f".{requested_name}")):
                continue
            matched_any = True
            if isinstance(module, supported_types):
                resolved.append(module_name)
                continue
            linear_child = getattr(module, "linear", None)
            if isinstance(linear_child, torch.nn.Linear):
                resolved.append(f"{module_name}.linear")
        if not matched_any:
            continue

    resolved = sorted(set(resolved))
    if not resolved:
        raise ValueError(
            f"None of the requested LoRA target modules were found: {requested}"
        )
    return resolved


class GemmaSftDataset(Dataset):
    def __init__(self, samples: list[dict[str, str]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.samples[idx]


@dataclass
class GemmaSftCollator:
    processor: Any
    sampling_rate: int
    max_seq_length: int
    max_prompt_length: int

    def __call__(self, features: list[dict[str, str]]) -> dict[str, torch.Tensor]:
        prompt_conversations = [build_prompt_messages(sample) for sample in features]
        full_conversations = [build_training_messages(sample) for sample in features]

        # For Gemma4 audio inputs, truncating the tokenized chat template can break the
        # alignment between audio placeholder tokens and extracted audio features.
        # Keep processor-side tokenization non-truncated for correctness.
        prompt_inputs = self.processor.apply_chat_template(
            prompt_conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            processor_kwargs={"sampling_rate": self.sampling_rate},
        )
        full_inputs = self.processor.apply_chat_template(
            full_conversations,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            processor_kwargs={"sampling_rate": self.sampling_rate},
        )

        # Gemma4Processor can emit input_features and input_features_mask with
        # off-by-a-few-frames mismatches in the time dimension when batch-padding
        # variable-length audio. The audio tower's layer0 then crashes on
        # `hidden_states * mask[:, None, :, None]`. Trim both sides to the
        # common length so they are guaranteed aligned.
        for inputs in (prompt_inputs, full_inputs):
            feats = inputs.get("input_features")
            feats_mask = inputs.get("input_features_mask")
            if feats is None or feats_mask is None:
                continue
            t_feats = feats.shape[-2]
            t_mask = feats_mask.shape[-1]
            if t_feats != t_mask:
                t_common = min(t_feats, t_mask)
                inputs["input_features"] = feats[..., :t_common, :].contiguous()
                inputs["input_features_mask"] = feats_mask[..., :t_common].contiguous()

        labels = full_inputs["input_ids"].clone()
        attention_mask = full_inputs["attention_mask"]
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=1).tolist()

        for row_index, prompt_length in enumerate(prompt_lengths):
            effective_prompt_length = min(int(prompt_length), labels.shape[1])
            labels[row_index, :effective_prompt_length] = -100
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
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class LastCheckpointTrackerCallback(TrainerCallback):
    def __init__(self) -> None:
        self.last_checkpoint_dir: Path | None = None

    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint_dir.exists():
            self.last_checkpoint_dir = checkpoint_dir


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def infer_device(model: torch.nn.Module) -> torch.device:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


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
    samples: list[dict[str, str]],
    processor,
    model,
    batch_size: int,
    sampling_rate: int,
    max_new_tokens: int,
    max_prompt_length: int,
) -> list[str]:
    predictions: list[str] = []
    device = infer_device(model)
    pad_token_id = get_pad_token_id(processor)
    eos_token_id = get_generation_stop_token_ids(processor)

    for start in range(0, len(samples), batch_size):
        batch = samples[start : start + batch_size]
        conversations = [build_prompt_messages(sample) for sample in batch]
        # Generation uses the same non-truncated multimodal prompt encoding as training
        # to avoid audio token / audio feature mismatches.
        model_inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            processor_kwargs={"sampling_rate": sampling_rate},
        )
        feats = model_inputs.get("input_features")
        feats_mask = model_inputs.get("input_features_mask")
        if feats is not None and feats_mask is not None and feats.shape[-2] != feats_mask.shape[-1]:
            t_common = min(feats.shape[-2], feats_mask.shape[-1])
            model_inputs["input_features"] = feats[..., :t_common, :].contiguous()
            model_inputs["input_features_mask"] = feats_mask[..., :t_common].contiguous()
        model_inputs = move_batch_to_device(model_inputs, device)

        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )

        prompt_length = model_inputs["input_ids"].shape[1]
        generated_only = generated[:, prompt_length:]
        decoded = processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        predictions.extend(normalize_text(text) for text in decoded)

    return predictions


def compute_generation_metrics(
    samples: list[dict[str, str]],
    processor,
    model,
    batch_size: int,
    sampling_rate: int,
    max_new_tokens: int,
    max_prompt_length: int,
) -> dict[str, float]:
    sacrebleu = get_sacrebleu()
    predictions = generate_predictions(
        samples=samples,
        processor=processor,
        model=model,
        batch_size=batch_size,
        sampling_rate=sampling_rate,
        max_new_tokens=max_new_tokens,
        max_prompt_length=max_prompt_length,
    )
    references = [sample["reference"] for sample in samples]
    bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="zh")
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=0)
    return {"bleu": float(bleu.score), "chrf": float(chrf.score)}


class GenerationMetricsCallback(TrainerCallback):
    def __init__(
        self,
        *,
        processor,
        train_samples: list[dict[str, str]],
        val_samples: list[dict[str, str]],
        config: argparse.Namespace,
    ) -> None:
        self.processor = processor
        self.train_samples = (
            train_samples[: config.metric_train_limit]
            if config.metric_train_limit is not None
            else train_samples
        )
        self.val_samples = (
            val_samples[: config.metric_val_limit]
            if config.metric_val_limit is not None
            else val_samples
        )
        self.config = config
        self.trainer: Trainer | None = None

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if self.trainer is None or not state.is_world_process_zero:
            return

        metric_row = {"step": state.global_step, "epoch": state.epoch}
        if self.train_samples:
            train_metrics = compute_generation_metrics(
                samples=self.train_samples,
                processor=self.processor,
                model=self.trainer.model,
                batch_size=self.config.per_device_eval_batch_size,
                sampling_rate=self.config.sampling_rate,
                max_new_tokens=self.config.metric_max_new_tokens,
                max_prompt_length=self.config.max_prompt_length
                or self.config.max_seq_length,
            )
            metric_row["train_bleu"] = train_metrics["bleu"]
            metric_row["train_chrf"] = train_metrics["chrf"]

        if self.val_samples:
            val_metrics = compute_generation_metrics(
                samples=self.val_samples,
                processor=self.processor,
                model=self.trainer.model,
                batch_size=self.config.per_device_eval_batch_size,
                sampling_rate=self.config.sampling_rate,
                max_new_tokens=self.config.metric_max_new_tokens,
                max_prompt_length=self.config.max_prompt_length
                or self.config.max_seq_length,
            )
            metric_row["val_bleu"] = val_metrics["bleu"]
            metric_row["val_chrf"] = val_metrics["chrf"]

        self.trainer.log(metric_row)


def load_model_and_processor(args: argparse.Namespace):
    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if torch.cuda.is_available() and local_rank >= 0:
        torch.cuda.set_device(local_rank)
    processor = AutoProcessor.from_pretrained(args.base_model_path)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=resolve_explicit_torch_dtype(
                args.bnb_4bit_compute_dtype
            ),
        )
    if torch.cuda.is_available() and local_rank >= 0:
        # In multi-process training each rank should warm up on its own local
        # CUDA device instead of letting every worker touch GPU 0 first.
        model_kwargs["device_map"] = {"": local_rank}
    elif args.device_map.lower() != "none":
        model_kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model_path, **model_kwargs
    )
    model.config.use_cache = False
    return processor, model


def apply_lora(model: torch.nn.Module, args: argparse.Namespace):
    try:
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError("peft is required for LoRA SFT training.") from exc

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )
    elif args.gradient_checkpointing:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    requested = get_requested_target_modules(args)
    target_modules = resolve_lora_target_modules(model, requested)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return model, target_modules, trainable_params, total_params


def copy_adapter_artifacts(source_dir: Path, target_dir: Path) -> bool:
    candidate_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "README.md",
    ]
    available = [source_dir / name for name in candidate_files if (source_dir / name).exists()]
    if not available:
        return False
    staging_dir = target_dir.with_name(f"{target_dir.name}.tmp")
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    for source in available:
        shutil.copyfile(source, staging_dir / source.name)
    if target_dir.exists():
        shutil.rmtree(target_dir, ignore_errors=True)
    staging_dir.replace(target_dir)
    return True


def save_adapter_atomically(model: torch.nn.Module, output_dir: Path) -> None:
    staging_dir = output_dir.with_name(f"{output_dir.name}.tmp")
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(staging_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir, ignore_errors=True)
    staging_dir.replace(output_dir)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_training_arguments(
    args: argparse.Namespace,
    trainer_output_dir: Path,
    tensorboard_dir: Path,
) -> TrainingArguments:
    use_steps = args.eval_steps > 0 or args.save_steps > 0
    effective_eval_steps = args.eval_steps if args.eval_steps > 0 else None
    effective_save_steps = args.save_steps if args.save_steps > 0 else None
    if use_steps:
        interval = effective_eval_steps or effective_save_steps or 200
        effective_eval_steps = interval
        effective_save_steps = interval
        evaluation_strategy = "steps"
        save_strategy = "steps"
    else:
        evaluation_strategy = "epoch"
        save_strategy = "epoch"

    report_to = ["tensorboard"] if args.enable_tensorboard else []
    signature = inspect.signature(TrainingArguments.__init__)
    training_kwargs: dict[str, Any] = {
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
    }
    if "overwrite_output_dir" in signature.parameters:
        training_kwargs["overwrite_output_dir"] = False
    if "evaluation_strategy" in signature.parameters:
        training_kwargs["evaluation_strategy"] = evaluation_strategy
    elif "eval_strategy" in signature.parameters:
        training_kwargs["eval_strategy"] = evaluation_strategy
    else:
        raise RuntimeError("TrainingArguments is missing evaluation/eval strategy parameter.")
    return TrainingArguments(**training_kwargs)


def main() -> None:
    args = parse_args()
    bind_local_cuda_device_from_env()
    seed_everything(args.seed)

    root_dir = Path(__file__).resolve().parents[1]
    experiment_dir = args.output_root / args.experiment_name
    adapter_best_dir = experiment_dir / "adapter_best"
    adapter_last_dir = experiment_dir / "adapter_last"
    trainer_output_dir = experiment_dir / "trainer_output"
    trainer_logs_path = experiment_dir / "trainer_logs.jsonl"
    run_config_path = experiment_dir / "run_config.json"
    train_state_path = experiment_dir / "train_state.json"
    tensorboard_dir = args.tensorboard_dir or (experiment_dir / "tensorboard")

    experiment_dir.mkdir(parents=True, exist_ok=True)

    if args.enable_tensorboard:
        try:
            import tensorboard  # noqa: F401
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TensorBoard logging requires tensorboard to be installed."
            ) from exc

    train_samples = load_samples(args.train_data_path, args, args.limit_train_samples)
    val_samples = load_samples(args.val_data_path, args, args.limit_val_samples)

    processor, model = load_model_and_processor(args)
    model, target_modules, trainable_params, total_params = apply_lora(model, args)

    prompt_max_length = args.max_prompt_length or args.max_seq_length
    data_collator = GemmaSftCollator(
        processor=processor,
        sampling_rate=args.sampling_rate,
        max_seq_length=args.max_seq_length,
        max_prompt_length=prompt_max_length,
    )

    training_args = build_training_arguments(args, trainer_output_dir, tensorboard_dir)
    world_size = max(int(os.environ.get("WORLD_SIZE", "1")), 1)
    global_micro_batch_size = (
        args.per_device_train_batch_size * world_size
    )
    effective_global_batch_size = (
        global_micro_batch_size * args.gradient_accumulation_steps
    )
    steps_per_epoch = math.ceil(len(train_samples) / max(global_micro_batch_size, 1))
    optimizer_steps_per_epoch = math.ceil(
        steps_per_epoch / max(args.gradient_accumulation_steps, 1)
    )

    run_config = {
        "experiment_name": args.experiment_name,
        "base_model_path": args.base_model_path,
        "train_data_path": str(args.train_data_path),
        "val_data_path": str(args.val_data_path),
        "test_data_path": str(args.test_data_path),
        "output_dir": str(experiment_dir),
        "audio_prefix_from": args.audio_prefix_from,
        "audio_prefix_to": args.audio_prefix_to,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "max_train_steps": args.max_train_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "max_grad_norm": args.max_grad_norm,
        "seed": args.seed,
        "sampling_rate": args.sampling_rate,
        "metric_max_new_tokens": args.metric_max_new_tokens,
        "max_seq_length": args.max_seq_length,
        "torch_dtype": args.torch_dtype,
        "mixed_precision": args.mixed_precision,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "gradient_checkpointing": args.gradient_checkpointing,
        "load_in_4bit": args.load_in_4bit,
        "enable_tensorboard": args.enable_tensorboard,
        "tensorboard_dir": str(tensorboard_dir),
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "save_total_limit": args.save_total_limit,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "limit_train_samples": args.limit_train_samples,
        "limit_val_samples": args.limit_val_samples,
        "metric_train_limit": args.metric_train_limit,
        "metric_val_limit": args.metric_val_limit,
        "optim": args.optim,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": target_modules,
        "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
        "bnb_4bit_compute_dtype": args.bnb_4bit_compute_dtype,
        "bnb_4bit_use_double_quant": args.bnb_4bit_use_double_quant,
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
        "world_size": world_size,
        "global_micro_batch_size": global_micro_batch_size,
        "effective_global_batch_size": effective_global_batch_size,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }
    if int(os.environ.get("RANK", "0")) == 0:
        save_json(run_config_path, run_config)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=GemmaSftDataset(train_samples),
        eval_dataset=GemmaSftDataset(val_samples),
        data_collator=data_collator,
        callbacks=[JsonlLoggingCallback(trainer_logs_path)],
    )
    last_checkpoint_tracker = LastCheckpointTrackerCallback()
    trainer.add_callback(last_checkpoint_tracker)
    generation_metrics_callback = GenerationMetricsCallback(
        processor=processor,
        train_samples=train_samples,
        val_samples=val_samples,
        config=args,
    )
    generation_metrics_callback.trainer = trainer
    trainer.add_callback(generation_metrics_callback)

    if trainer.is_world_process_zero():
        print(
            json.dumps(
                {
                    "experiment_name": args.experiment_name,
                    "train_samples": len(train_samples),
                    "val_samples": len(val_samples),
                    "world_size": world_size,
                    "global_micro_batch_size": global_micro_batch_size,
                    "effective_global_batch_size": effective_global_batch_size,
                    "steps_per_epoch": steps_per_epoch,
                    "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
                    "target_modules": target_modules,
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "trainer_output_dir": str(trainer_output_dir),
                },
                ensure_ascii=False,
            )
        )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        trainer.state.save_to_json(str(train_state_path))
    final_eval_metrics = trainer.evaluate()

    trainer.accelerator.wait_for_everyone()

    last_checkpoint = (
        str(last_checkpoint_tracker.last_checkpoint_dir)
        if last_checkpoint_tracker.last_checkpoint_dir is not None
        else get_last_checkpoint(str(trainer_output_dir))
    )
    copied_last = False
    copied_best = False
    if trainer.is_world_process_zero():
        if last_checkpoint:
            copied_last = copy_adapter_artifacts(Path(last_checkpoint), adapter_last_dir)

        if trainer.state.best_model_checkpoint:
            copied_best = copy_adapter_artifacts(
                Path(trainer.state.best_model_checkpoint), adapter_best_dir
            )
        if not copied_best and copied_last:
            copied_best = copy_adapter_artifacts(adapter_last_dir, adapter_best_dir)
        if not copied_best:
            save_adapter_atomically(trainer.model, adapter_best_dir)

    trainer.accelerator.wait_for_everyone()

    metrics = train_result.metrics
    metrics.update({f"final_{key}": value for key, value in final_eval_metrics.items()})
    metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint
    metrics["last_checkpoint"] = last_checkpoint
    metrics["adapter_last_available"] = copied_last
    if trainer.is_world_process_zero():
        save_json(experiment_dir / "train_result.json", metrics)

    if trainer.is_world_process_zero():
        eval_command = (
            "python scripts/eval_testt_bleu_chrf.py "
            f"--base-model-path {args.base_model_path} "
            f"--adapter-path {adapter_best_dir} "
            f"--data-path {args.test_data_path}"
        )
        print(json.dumps({"adapter_best": str(adapter_best_dir)}, ensure_ascii=False))
        print(
            json.dumps(
                {
                    "adapter_last": str(adapter_last_dir) if copied_last else None,
                },
                ensure_ascii=False,
            )
        )
        print(json.dumps({"eval_command": eval_command}, ensure_ascii=False))


if __name__ == "__main__":
    main()

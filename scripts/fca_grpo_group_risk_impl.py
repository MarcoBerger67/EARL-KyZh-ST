from __future__ import annotations

import argparse
import contextlib
import functools
import json
import logging
import math
import os
import random
import shutil
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
    TorchAoConfig,
    get_scheduler,
)
from transformers.pytorch_utils import Conv1D


def get_summary_writer():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except (ModuleNotFoundError, ImportError):
        return None
    return SummaryWriter


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
PLACEHOLDER_TRANSLATION = "<Simplified Chinese translation>"
DEFAULT_BASE_MODEL_PATH = "gemma-4-E4B-it"
DEFAULT_LORA_TARGET_MODULES = (
    "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)

warnings.filterwarnings(
    "ignore",
    message=r"Kwargs passed to `processor\.__call__` have to be in `processor_kwargs` dict, not in `\*\*kwargs`",
)

PROCESSOR_KWARGS_NOISE = (
    "Kwargs passed to `processor.__call__` have to be in `processor_kwargs` dict"
)


class _AudioTextOnlyGemma4Processor:
    """Small Gemma4 audio/text processor fallback for older Transformers builds."""

    def __init__(self, tokenizer: Any, feature_extractor: Any):
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.audio_token = getattr(tokenizer, "audio_token", "<|audio|>")
        self.audio_token_id = getattr(tokenizer, "audio_token_id", None)
        if self.audio_token_id is None:
            self.audio_token_id = tokenizer.convert_tokens_to_ids(self.audio_token)

    @classmethod
    def from_pretrained(cls, model_path: str | os.PathLike[str]) -> "_AudioTextOnlyGemma4Processor":
        try:
            from transformers import Gemma3nAudioFeatureExtractor
        except ImportError as exc:
            raise RuntimeError(
                "AutoProcessor failed, and this Transformers build also lacks "
                "Gemma3nAudioFeatureExtractor for the audio/text fallback."
            ) from exc

        model_dir = Path(model_path)
        config_path = model_dir / "processor_config.json"
        processor_config: dict[str, Any] = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as fh:
                processor_config = json.load(fh)
        tokenizer_config_path = model_dir / "tokenizer_config.json"
        tokenizer_config: dict[str, Any] = {}
        if tokenizer_config_path.exists():
            with tokenizer_config_path.open("r", encoding="utf-8") as fh:
                tokenizer_config = json.load(fh)
        feature_config = dict(processor_config.get("feature_extractor") or {})
        sampling_rate = int(feature_config.get("sampling_rate", 16000))

        def _samples_to_ms(key: str, default_ms: float) -> float:
            value = feature_config.get(key)
            if value is None:
                return default_ms
            return float(value) / float(sampling_rate) * 1000.0

        feature_extractor = Gemma3nAudioFeatureExtractor(
            feature_size=int(feature_config.get("feature_size", 128)),
            sampling_rate=sampling_rate,
            padding_value=float(feature_config.get("padding_value", 0.0)),
            return_attention_mask=bool(feature_config.get("return_attention_mask", True)),
            frame_length_ms=_samples_to_ms("frame_length", 20.0),
            hop_length_ms=_samples_to_ms("hop_length", 10.0),
            min_frequency=float(feature_config.get("min_frequency", 0.0)),
            max_frequency=float(feature_config.get("max_frequency", 8000.0)),
            preemphasis=float(feature_config.get("preemphasis", 0.0)),
            preemphasis_htk_flavor=bool(feature_config.get("preemphasis_htk_flavor", True)),
            fft_overdrive=bool(feature_config.get("fft_overdrive", False)),
            dither=float(feature_config.get("dither", 0.0)),
            input_scale_factor=float(feature_config.get("input_scale_factor", 1.0)),
            mel_floor=float(feature_config.get("mel_floor", 0.001)),
            per_bin_mean=feature_config.get("per_bin_mean"),
            per_bin_stddev=feature_config.get("per_bin_stddev"),
        )
        tokenizer = cls._load_tokenizer(model_dir, tokenizer_config)
        return cls(tokenizer=tokenizer, feature_extractor=feature_extractor)

    @staticmethod
    def _load_tokenizer(model_dir: Path, tokenizer_config: dict[str, Any]) -> Any:
        try:
            return AutoTokenizer.from_pretrained(
                model_dir,
                trust_remote_code=True,
                use_fast=False,
            )
        except (AttributeError, TypeError, ValueError):
            pass

        from transformers import PreTrainedTokenizerFast

        tokenizer_file = model_dir / "tokenizer.json"
        if not tokenizer_file.exists():
            raise RuntimeError(
                f"Cannot load Gemma4 fallback tokenizer: missing {tokenizer_file}."
            )

        special_token_kwargs: dict[str, Any] = {}
        additional_special_tokens: list[str] = []
        for key, value in tokenizer_config.items():
            if key in {"bos_token", "eos_token", "unk_token", "pad_token", "mask_token"}:
                special_token_kwargs[key] = value
            elif key.endswith("_token") and isinstance(value, str):
                additional_special_tokens.append(value)
        extra_special_tokens = tokenizer_config.get("extra_special_tokens")
        if isinstance(extra_special_tokens, list):
            additional_special_tokens.extend(
                token for token in extra_special_tokens if isinstance(token, str)
            )
        if additional_special_tokens:
            special_token_kwargs["additional_special_tokens"] = sorted(
                set(additional_special_tokens)
            )

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(tokenizer_file),
            **special_token_kwargs,
        )
        tokenizer.padding_side = str(tokenizer_config.get("padding_side", "left"))
        if "model_max_length" in tokenizer_config:
            tokenizer.model_max_length = int(tokenizer_config["model_max_length"])
        chat_template_path = model_dir / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer.chat_template = chat_template_path.read_text(encoding="utf-8")
        elif isinstance(tokenizer_config.get("chat_template"), str):
            tokenizer.chat_template = tokenizer_config["chat_template"]

        for key, value in tokenizer_config.items():
            if key.endswith("_token") and isinstance(value, str):
                setattr(tokenizer, key, value)
                token_id = tokenizer.convert_tokens_to_ids(value)
                if isinstance(token_id, int) and token_id >= 0:
                    setattr(tokenizer, f"{key}_id", token_id)
        return tokenizer

    def batch_decode(self, *args: Any, **kwargs: Any) -> list[str]:
        return self.tokenizer.batch_decode(*args, **kwargs)

    def apply_chat_template(
        self,
        conversations: Any,
        *args: Any,
        tokenize: bool = False,
        return_dict: bool = False,
        return_tensors: str | None = None,
        padding: bool | str = False,
        processor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if not tokenize:
            return self.tokenizer.apply_chat_template(
                conversations, *args, tokenize=False, **kwargs
            )
        if not isinstance(conversations, list) or (
            conversations and isinstance(conversations[0], dict)
        ):
            conversation_batch = [conversations]
        else:
            conversation_batch = conversations

        texts = [
            self.tokenizer.apply_chat_template(
                conversation, *args, tokenize=False, **kwargs
            )
            for conversation in conversation_batch
        ]
        tokenized = self.tokenizer(
            texts,
            return_tensors=return_tensors,
            padding=padding,
            add_special_tokens=False,
        )
        audios = self._extract_audio_arrays(conversation_batch)
        if audios:
            audio_features = self.feature_extractor(
                audios,
                return_tensors=return_tensors,
                padding=True,
                **(processor_kwargs or {}),
            )
            tokenized.update(audio_features)
            if return_tensors == "pt":
                self._expand_audio_tokens_pt(tokenized)
        if return_dict:
            return tokenized
        return tokenized["input_ids"]

    @staticmethod
    def _extract_audio_arrays(conversations: list[list[dict[str, Any]]]) -> list[Any]:
        audios: list[Any] = []
        for conversation in conversations:
            for message in conversation:
                for item in message.get("content", []):
                    if isinstance(item, dict) and item.get("type") == "audio":
                        audio = item.get("audio")
                        if audio is not None:
                            audios.append(audio)
        return audios

    def _expand_audio_tokens_pt(self, batch: dict[str, Any]) -> None:
        input_ids = batch.get("input_ids")
        attention_mask = batch.get("attention_mask")
        features_mask = batch.get("input_features_mask")
        if not (
            isinstance(input_ids, torch.Tensor)
            and isinstance(attention_mask, torch.Tensor)
            and isinstance(features_mask, torch.Tensor)
        ):
            return
        expected_counts = estimate_audio_soft_token_counts(features_mask).to(input_ids.device)
        if expected_counts.numel() != input_ids.size(0):
            return
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = 0
        new_input_rows: list[torch.Tensor] = []
        new_mask_rows: list[torch.Tensor] = []
        for row_index, row in enumerate(input_ids):
            mask_row = attention_mask[row_index]
            expected_count = int(expected_counts[row_index].item())
            expanded_ids: list[torch.Tensor] = []
            expanded_mask: list[torch.Tensor] = []
            for token, mask_value in zip(row, mask_row):
                repeat = expected_count if int(token.item()) == int(self.audio_token_id) else 1
                expanded_ids.append(token.repeat(repeat))
                expanded_mask.append(mask_value.repeat(repeat))
            new_input_rows.append(torch.cat(expanded_ids))
            new_mask_rows.append(torch.cat(expanded_mask))
        batch["input_ids"] = _pad_trimmed_sequence_rows(new_input_rows, int(pad_id), True)
        batch["attention_mask"] = _pad_trimmed_sequence_rows(new_mask_rows, 0, True)


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
            "sacrebleu is required for reward computation and evaluation. "
            "Install it with `pip install sacrebleu`."
        ) from exc
    return sacrebleu


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train Gemma4 with LoRA-based group-relative optimization from an SFT checkpoint."
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default=DEFAULT_BASE_MODEL_PATH,
        help="Local or remote Hugging Face base model directory.",
    )
    parser.add_argument(
        "--init-adapter-path",
        type=Path,
        default=None,
        help="Optional SFT/LoRA adapter checkpoint used as the policy starting point.",
    )
    parser.add_argument(
        "--train-data-path",
        type=Path,
        default=(
            root_dir
            / "data"
            / "converted_testt_format"
            / "train_ky2zh_full285h_stage1.cleaned.jsonl"
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
        "--train-entity-path",
        type=Path,
        default=None,
        help="Optional JSONL sidecar containing entities/constraints/keywords for train samples.",
    )
    parser.add_argument(
        "--val-entity-path",
        type=Path,
        default=None,
        help="Optional JSONL sidecar containing entities/constraints/keywords for validation samples.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root_dir / "model" / "grpo",
        help="Root directory for group-relative optimization outputs.",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional override for the experiment directory name.",
    )
    parser.add_argument(
        "--enable-tensorboard",
        action="store_true",
        help="Write TensorBoard logs under the experiment directory.",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=Path,
        default=None,
        help="Optional override for the TensorBoard log directory. Defaults to <experiment_dir>/tensorboard.",
    )
    parser.add_argument(
        "--objective",
        choices=["weighted_sum", "chrf_only", "bleu_only", "chrf_bleu"],
        default="weighted_sum",
        help="Objective compatibility mode. Use weighted_sum for the default group-relative objective.",
    )
    parser.add_argument("--bleu-weight", type=float, default=0.0)
    parser.add_argument("--chrf-weight", type=float, default=0.0)
    parser.add_argument(
        "--entity-weight",
        dest="key_weight",
        type=float,
        default=None,
        help="Alias for --key-weight. Used by Stage II entity fidelity rewards.",
    )
    parser.add_argument("--key-weight", type=float, default=0.0)
    parser.add_argument("--ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--entity-reward-mode",
        choices=["none", "key_recall", "entity_em", "entity_soft", "entity_gemma_fuzzy", "entity_substring"],
        default="key_recall",
        help="Entity fidelity reward used when the entity/key weight is non-zero.",
    )
    parser.add_argument("--entity-soft-tau", type=float, default=0.6)
    parser.add_argument(
        "--entity-include-per",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include PER (person name) entities in the entity substring/fuzzy reward. "
             "Enabled by default (covers all entity types). Pass --no-entity-include-per "
             "to exclude PER, e.g. because Ky→Zh person names are phonetic transliterations.",
    )
    parser.add_argument("--entity-embedding-model", type=str, default="bert-base-multilingual-cased")
    parser.add_argument("--entity-embedding-max-length", type=int, default=128)
    parser.add_argument("--entity-embedding-pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument(
        "--entity-embedding-device",
        type=str,
        default="cpu",
        help="Device for the mBERT entity embedder ('cpu' or 'cuda'). Default cpu to avoid sharing the policy GPU.",
    )
    parser.add_argument(
        "--entity-extractor-device",
        type=str,
        default="cpu",
        help="Device for the HanLP entity extractor ('cpu' or 'cuda'). Default cpu to avoid sharing the policy GPU.",
    )
    parser.add_argument("--ner-tokenizer-model", type=str, default="FINE_ELECTRA_SMALL_ZH")
    parser.add_argument("--ner-model", type=str, default="MSRA_NER_ELECTRA_SMALL_ZH")
    parser.add_argument("--ner-tokenizer-path", type=str, default=None)
    parser.add_argument("--ner-path", type=str, default=None)
    parser.add_argument(
        "--gemma-ner-model-path",
        type=str,
        default=None,
        help="Path to a Gemma model used for entity extraction in entity_gemma_fuzzy mode.",
    )
    parser.add_argument("--audio-prefix-from", type=str, default="")
    parser.add_argument("--audio-prefix-to", type=str, default="")
    parser.add_argument("--num-train-epochs", type=lambda x: int(float(x)), default=1)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
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
        "--group-policy-scale",
        type=float,
        default=0.005,
        help="Scale applied to candidate log probabilities before the group softmax.",
    )
    parser.add_argument(
        "--kl-coef",
        type=float,
        default=0.0,
        help=(
            "Optional KL regularization coefficient. When > 0, the group objective adds "
            "kl_coef * KL(policy_candidate_distribution || reference_adapter_candidate_distribution)."
        ),
    )
    parser.add_argument(
        "--generation-strategy",
        choices=["sample", "beam", "beam_sample"],
        default="beam_sample",
        help=(
            "Candidate generation method during group-relative optimization. "
            "'beam_sample' (default) — beam search + per-step multinomial sampling, "
            "robust on low-entropy SFT models, no multilingual leakage. "
            "'beam' — plain beam search (top-K hypotheses, candidates may be similar). "
            "'sample' — pure stochastic top-k/top-p sampling (fragile)."
        ),
    )
    parser.add_argument("--diversity-penalty", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.2)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--val-max-new-tokens", type=int, default=None)
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
        default="no",
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
    parser.add_argument("--eval-every-steps", type=int, default=200)
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
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default=DEFAULT_LORA_TARGET_MODULES,
    )
    parser.add_argument(
        "--grpo-objective",
        "--objective-variant",
        choices=["group_relative_risk_kl", "clipped_grpo"],
        default="group_relative_risk_kl",
        help=(
            "Public GRPO objective label. The default runtime implements "
            "group_relative_risk_kl; clipped_grpo is handled by fca_grpo_clipped_runtime."
        ),
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=None,
        help="Compatibility no-op for clipped GRPO configs; ignored by group_relative_risk_kl.",
    )
    parser.add_argument(
        "--score-batch-size",
        type=int,
        default=None,
        help="Compatibility no-op for clipped GRPO configs; scoring is internally batched.",
    )
    parser.add_argument(
        "--disable-kl-monitor",
        action="store_true",
        help="Compatibility no-op for clipped GRPO configs.",
    )
    parser.add_argument(
        "--train-lora-module-filter",
        type=str,
        default="",
        help="Compatibility no-op for clipped GRPO configs.",
    )
    parser.add_argument(
        "--train-projector",
        action="store_true",
        help="Compatibility no-op for clipped GRPO configs.",
    )
    parser.add_argument(
        "--projector-module-filter",
        type=str,
        default="",
        help="Compatibility no-op for clipped GRPO configs.",
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


def patch_torch_finfo_for_quantized_gemma4(args: argparse.Namespace) -> None:
    if not (args.load_in_4bit or args.load_in_8bit or args.torchao_int8_weight_only):
        return
    if getattr(torch, "_gemma4_quant_finfo_patched", False):
        return

    original_finfo = torch.finfo

    def safe_finfo(dtype):
        try:
            return original_finfo(dtype)
        except TypeError:
            if isinstance(dtype, torch.dtype) and not dtype.is_floating_point:
                return original_finfo(torch.float32)
            raise

    torch.finfo = safe_finfo
    torch._gemma4_quant_finfo_patched = True


def patch_torch_masked_scatter_for_quantized_gemma4(args: argparse.Namespace) -> None:
    if not (args.load_in_4bit or args.load_in_8bit or args.torchao_int8_weight_only):
        return
    if getattr(torch, "_gemma4_quant_masked_scatter_patched", False):
        return

    original_masked_scatter = torch.Tensor.masked_scatter
    original_masked_scatter_ = torch.Tensor.masked_scatter_

    def safe_masked_scatter(self, mask, source):
        target = self
        if (
            isinstance(self, torch.Tensor)
            and isinstance(source, torch.Tensor)
            and torch.is_floating_point(self)
            and torch.is_floating_point(source)
            and self.dtype != source.dtype
        ):
            target = self.to(source.dtype)
        return original_masked_scatter(target, mask, source)

    def safe_masked_scatter_(self, mask, source):
        if (
            isinstance(self, torch.Tensor)
            and isinstance(source, torch.Tensor)
            and torch.is_floating_point(self)
            and torch.is_floating_point(source)
            and self.dtype != source.dtype
        ):
            converted = self.to(source.dtype)
            return original_masked_scatter_(converted, mask, source)
        return original_masked_scatter_(self, mask, source)

    torch.Tensor.masked_scatter = safe_masked_scatter
    torch.Tensor.masked_scatter_ = safe_masked_scatter_
    torch._gemma4_quant_masked_scatter_patched = True


def patch_torch_masked_fill_for_low_precision(args: argparse.Namespace) -> None:
    if args.mixed_precision not in {"fp16", "bf16"}:
        return
    if getattr(torch, "_gemma4_low_precision_masked_fill_patched", False):
        return

    original_masked_fill = torch.Tensor.masked_fill
    original_masked_fill_ = torch.Tensor.masked_fill_

    def _normalize_fill_value(tensor: torch.Tensor, value: Any) -> Any:
        if not isinstance(tensor, torch.Tensor) or not torch.is_floating_point(tensor):
            return value
        if not isinstance(value, (float, int)):
            return value
        finfo = torch.finfo(tensor.dtype)
        clamped = min(max(float(value), finfo.min), finfo.max)
        return clamped

    def safe_masked_fill(self, mask, value):
        return original_masked_fill(self, mask, _normalize_fill_value(self, value))

    def safe_masked_fill_(self, mask, value):
        return original_masked_fill_(self, mask, _normalize_fill_value(self, value))

    torch.Tensor.masked_fill = safe_masked_fill
    torch.Tensor.masked_fill_ = safe_masked_fill_
    torch._gemma4_low_precision_masked_fill_patched = True


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = text.strip()
    if text.startswith("Translation:"):
        text = text[len("Translation:") :].strip()
    return " ".join(text.split())


def normalize_key_text(text: str) -> str:
    return normalize_text(text).replace(" ", "").lower()


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
    # for evaluation and RL rollout.
    return DEFAULT_PROMPT


def extract_key_texts_from_record(record: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    entities = record.get("entities")
    if isinstance(entities, list):
        for entity in entities:
            text = entity.get("text") if isinstance(entity, dict) else entity
            if text:
                keys.append(str(text))

    for field_name in ("constraints", "keywords"):
        values = record.get(field_name)
        if not isinstance(values, list):
            continue
        for item in values:
            if isinstance(item, dict):
                text = item.get("text") or item.get("value") or item.get("keyword")
            else:
                text = item
            if text:
                keys.append(str(text))

    normalized = [normalize_key_text(text) for text in keys if normalize_key_text(text)]
    return sorted(set(normalized))


def extract_entities_from_record(record: dict[str, Any]) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    raw_entities = record.get("entities")
    if not isinstance(raw_entities, list):
        return entities
    seen: set[tuple[str, str]] = set()
    for item in raw_entities:
        if isinstance(item, dict):
            text = str(item.get("text", "") or "").strip()
            label = str(item.get("label", "") or item.get("type", "") or "MISC").strip().upper()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            text = str(item[0] or "").strip()
            label = str(item[1] or "MISC").strip().upper()
        else:
            text = str(item or "").strip()
            label = "MISC"
        if not text:
            continue
        key = (text, label)
        if key in seen:
            continue
        seen.add(key)
        entities.append({"text": text, "label": label})
    return entities


def parse_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
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
            "gold_keys": extract_key_texts_from_record(record),
            "gold_entities": extract_entities_from_record(record),
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
            "gold_keys": extract_key_texts_from_record(record),
            "gold_entities": extract_entities_from_record(record),
        }

    raise ValueError(
        "Unsupported input record format. Expected either key/audio/gt or id/messages."
    )


def load_samples(
    data_path: Path, args: argparse.Namespace, limit: int | None
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with data_path.open("r", encoding="utf-8") as f:
        for source_index, line in enumerate(f, start=1):
            if not line.strip():
                continue
            sample = parse_record(json.loads(line), args)
            if sample is None:
                continue
            sample["source_index"] = source_index
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise ValueError(f"No valid samples were loaded from {data_path}.")
    return samples


def load_key_sidecar(path: Path | None) -> dict[str, list[str]]:
    if path is None:
        return {}
    sidecar: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = record.get("id") or record.get("key")
            if record_id:
                sidecar[str(record_id)] = extract_key_texts_from_record(record)
    return sidecar


def load_entity_sidecar(path: Path | None) -> dict[str, list[dict[str, str]]]:
    if path is None:
        return {}
    sidecar: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            record_id = record.get("id") or record.get("key")
            if record_id:
                sidecar[str(record_id)] = extract_entities_from_record(record)
    return sidecar


def attach_sidecar_keys(
    samples: list[dict[str, Any]],
    key_map: dict[str, list[str]],
) -> list[dict[str, Any]]:
    if not key_map:
        return [{**sample, "gold_keys": sorted(set(sample.get("gold_keys", [])))} for sample in samples]

    merged_samples: list[dict[str, Any]] = []
    for sample in samples:
        merged_keys = set(sample.get("gold_keys", []))
        merged_keys.update(key_map.get(sample["id"], []))
        merged_samples.append({**sample, "gold_keys": sorted(merged_keys)})
    return merged_samples


def attach_sidecar_annotations(
    samples: list[dict[str, Any]],
    key_map: dict[str, list[str]],
    entity_map: dict[str, list[dict[str, str]]] | None = None,
) -> list[dict[str, Any]]:
    entity_map = entity_map or {}
    with_keys = attach_sidecar_keys(samples, key_map)
    annotated: list[dict[str, Any]] = []
    for sample in with_keys:
        merged_entities = list(sample.get("gold_entities", []))
        merged_entities.extend(entity_map.get(sample["id"], []))
        deduped_entities: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for entity in merged_entities:
            if not isinstance(entity, dict):
                continue
            text = str(entity.get("text", "") or "").strip()
            label = str(entity.get("label", "") or "MISC").strip().upper()
            if not text:
                continue
            key = (text, label)
            if key in seen:
                continue
            seen.add(key)
            deduped_entities.append({"text": text, "label": label})
        annotated.append({**sample, "gold_entities": deduped_entities})
    return annotated


def resolve_weight_config(args: argparse.Namespace) -> dict[str, Any]:
    key_weight = 0.0 if args.key_weight is None else args.key_weight
    if args.objective == "bleu_only":
        raw_weights = {"bleu": 1.0, "chrf": 0.0, "key": 0.0, "ce": 0.0}
    elif args.objective == "chrf_only":
        raw_weights = {"bleu": 0.0, "chrf": 1.0, "key": 0.0, "ce": 0.0}
    elif args.objective == "chrf_bleu":
        if key_weight > 0 or args.ce_weight > 0:
            raise ValueError(
                "--objective chrf_bleu is only compatible with BLEU/chrF weights. "
                "Use --objective weighted_sum for the group-relative objective."
            )
        raw_weights = {
            "bleu": args.bleu_weight,
            "chrf": args.chrf_weight,
            "key": 0.0,
            "ce": 0.0,
        }
    else:
        raw_weights = {
            "bleu": args.bleu_weight,
            "chrf": args.chrf_weight,
            "key": key_weight,
            "ce": args.ce_weight,
        }

    for name, value in raw_weights.items():
        if value < 0:
            raise ValueError(f"{name} weight must be >= 0.")
    total = sum(raw_weights.values())
    if total <= 0:
        raise ValueError("At least one of BLEU/chrF/key/CE weights must be > 0.")

    normalized = {name: value / total for name, value in raw_weights.items()}
    risk_objective_weight = normalized["bleu"] + normalized["chrf"] + normalized["key"]
    if risk_objective_weight > 0:
        reward_weights = {
            "bleu": normalized["bleu"] / risk_objective_weight,
            "chrf": normalized["chrf"] / risk_objective_weight,
            "key": normalized["key"] / risk_objective_weight,
        }
    else:
        reward_weights = {"bleu": 0.0, "chrf": 0.0, "key": 0.0}

    return {
        "raw": raw_weights,
        "normalized": normalized,
        "reward_weights": reward_weights,
        "risk_objective_weight": risk_objective_weight,
        "ce_weight": normalized["ce"],
        "key_match_mode": args.key_match_mode,
        "entity_reward_mode": args.entity_reward_mode,
        "entity_soft_tau": args.entity_soft_tau,
        "entity_embedding_model": args.entity_embedding_model,
        "entity_embedding_pooling": args.entity_embedding_pooling,
        "gemma_ner_model_path": getattr(args, "gemma_ner_model_path", None),
        "effective_gradient_checkpointing": should_enable_model_gradient_checkpointing(args),
    }


def format_float_component(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    if "." not in text:
        text = f"{text}.0"
    return text.replace(".", "p")


def build_experiment_name(weight_config: dict[str, Any]) -> str:
    normalized = weight_config["normalized"]
    return (
        "group_risk_"
        f"b{format_float_component(normalized['bleu'])}_"
        f"c{format_float_component(normalized['chrf'])}_"
        f"k{format_float_component(normalized['key'])}_"
        f"ce{format_float_component(normalized['ce'])}"
    )


def iter_batches(samples: list[dict[str, Any]], batch_size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(samples), batch_size):
        yield samples[start : start + batch_size]


def _resample_audio_np(
    audio: np.ndarray, source_rate: int, target_rate: int
) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return audio.astype(np.float32, copy=False)
    duration = audio.shape[0] / float(source_rate)
    target_length = max(int(round(duration * float(target_rate))), 1)
    source_positions = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False)
    resampled = np.interp(target_positions, source_positions, audio)
    return resampled.astype(np.float32, copy=False)


def _normalize_audio_array(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio)
    audio = np.squeeze(audio)
    if audio.ndim == 0:
        audio = audio.reshape(1)
    elif audio.ndim == 2:
        # Normalize both [time, channels] and [channels, time] layouts to mono.
        if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=1)
    elif audio.ndim > 2:
        audio = audio.reshape(-1)
    return np.asarray(audio, dtype=np.float32)


@functools.lru_cache(maxsize=16384)
def load_audio_array(audio_path: str, target_sampling_rate: int) -> np.ndarray:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    last_error: Exception | None = None

    try:
        import soundfile as sf

        audio, sampling_rate = sf.read(str(path), always_2d=False)
        audio = _normalize_audio_array(audio)
        return _resample_audio_np(audio, int(sampling_rate), int(target_sampling_rate))
    except Exception as exc:
        last_error = exc

    try:
        import librosa

        audio, _ = librosa.load(
            str(path),
            sr=int(target_sampling_rate),
            mono=True,
        )
        return _normalize_audio_array(audio)
    except Exception as exc:
        last_error = exc

    raise RuntimeError(
        f"Failed to load audio from {audio_path} with supported backends."
    ) from last_error


_OVERRIDE_PROMPT_CACHE: dict[str, str | None] = {"value": None, "loaded": False}


def _resolve_override_prompt() -> str | None:
    if _OVERRIDE_PROMPT_CACHE["loaded"]:
        return _OVERRIDE_PROMPT_CACHE["value"]
    text = os.environ.get("OVERRIDE_PROMPT_TEXT")
    if not text:
        path = os.environ.get("OVERRIDE_PROMPT_FILE")
        if path:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    text = fh.read().strip()
            except OSError:
                text = None
    _OVERRIDE_PROMPT_CACHE["value"] = text
    _OVERRIDE_PROMPT_CACHE["loaded"] = True
    if text:
        print(
            json.dumps(
                {
                    "_debug": "override_prompt_active",
                    "length": len(text),
                    "head": text[:200],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    return text


def build_prompt_messages(
    sample: dict[str, Any], sampling_rate: int
) -> list[dict[str, Any]]:
    audio_array = load_audio_array(sample["audio_path"], sampling_rate)
    prompt_text = _resolve_override_prompt() or sample["prompt"]
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio_array},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]


def get_runtime_float_dtype(args: argparse.Namespace) -> torch.dtype | None:
    if args.mixed_precision == "bf16":
        return torch.bfloat16
    if args.mixed_precision == "fp16":
        return torch.float16
    return None


def move_batch_to_device(
    batch: dict[str, Any],
    device: torch.device,
    float_dtype: torch.dtype | None = None,
    audio_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> dict[str, Any]:
    batch = align_audio_feature_mask(batch)
    batch = trim_surplus_audio_tokens(batch, audio_token_id, pad_token_id)
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if not isinstance(value, torch.Tensor):
            moved[key] = value
            continue
        tensor = value.to(device)
        if float_dtype is not None and torch.is_floating_point(tensor):
            tensor = tensor.to(float_dtype)
        moved[key] = tensor
    return moved


def ensure_tensor_has_batch_dim(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and value.ndim == 1:
        return value.unsqueeze(0)
    return value


def ensure_batch_dims(batch: dict[str, Any]) -> dict[str, Any]:
    return {key: ensure_tensor_has_batch_dim(value) for key, value in batch.items()}


def align_audio_feature_mask(batch: dict[str, Any]) -> dict[str, Any]:
    features = batch.get("input_features")
    mask = batch.get("input_features_mask")
    if not isinstance(features, torch.Tensor) or not isinstance(mask, torch.Tensor):
        return batch
    if features.ndim < 2 or mask.ndim < 1:
        return batch

    # Gemma4's audio convolution multiplies hidden states by
    # mask[:, None, :, None]. Some processor outputs can be off by a few frames
    # for boundary-length audio, so align the mask to the feature time axis.
    expected_length = features.shape[-2] if features.ndim >= 3 else features.shape[-1]
    current_length = mask.shape[-1]
    if current_length == expected_length:
        return batch

    aligned = dict(batch)
    if current_length > expected_length:
        aligned["input_features_mask"] = mask[..., :expected_length]
    else:
        aligned["input_features_mask"] = F.pad(
            mask,
            (0, expected_length - current_length),
            value=0,
        )
    return aligned


def get_audio_token_id(processor: Any) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    direct_id = getattr(processor, "audio_token_id", None)
    if direct_id is None and tokenizer is not None:
        direct_id = getattr(tokenizer, "audio_token_id", None)
    if direct_id is not None:
        return int(direct_id)
    token_candidates = [
        getattr(processor, "audio_token", None),
        getattr(tokenizer, "audio_token", None) if tokenizer is not None else None,
        "<|audio|>",
    ]
    for token in token_candidates:
        if not token or tokenizer is None or not hasattr(tokenizer, "convert_tokens_to_ids"):
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        if isinstance(token_id, int) and token_id >= 0 and (unk_id is None or token_id != unk_id):
            return token_id
    return None


def estimate_audio_soft_token_counts(input_features_mask: torch.Tensor) -> torch.Tensor:
    mask = input_features_mask.bool()
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    # Gemma4AudioSubSampleConvProjection applies two stride-2 conv layers and
    # downsamples the mask with mask[:, ::2] after each layer.
    mask = mask[..., ::2]
    mask = mask[..., ::2]
    return mask.reshape(-1, mask.shape[-1]).sum(dim=-1).to(torch.long)


def _expand_audio_counts(counts: torch.Tensor, batch_size: int) -> torch.Tensor | None:
    if counts.numel() == batch_size:
        return counts
    if counts.numel() == 1:
        return counts.expand(batch_size)
    if batch_size % counts.numel() == 0:
        repeats = batch_size // counts.numel()
        return counts.repeat_interleave(repeats)
    return None


def _pad_trimmed_sequence_rows(
    rows: list[torch.Tensor],
    pad_value: int,
    left_pad: bool,
) -> torch.Tensor:
    max_length = max(row.size(-1) for row in rows)
    padded_rows = []
    for row in rows:
        pad_length = max_length - row.size(-1)
        if pad_length <= 0:
            padded_rows.append(row)
            continue
        padding = row.new_full((pad_length,), pad_value)
        padded_rows.append(torch.cat([padding, row], dim=0) if left_pad else torch.cat([row, padding], dim=0))
    return torch.stack(padded_rows, dim=0)


def trim_surplus_audio_tokens(
    batch: dict[str, Any],
    audio_token_id: int | None,
    pad_token_id: int | None = None,
) -> dict[str, Any]:
    if audio_token_id is None:
        return batch
    input_ids = batch.get("input_ids")
    input_features_mask = batch.get("input_features_mask")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(input_features_mask, torch.Tensor):
        return batch
    if input_ids.ndim != 2:
        return batch

    expected_counts = _expand_audio_counts(
        estimate_audio_soft_token_counts(input_features_mask).to(device=input_ids.device),
        input_ids.size(0),
    )
    if expected_counts is None:
        return batch

    keep_masks: list[torch.Tensor] = []
    needs_trim = False
    for row_index, row in enumerate(input_ids):
        audio_positions = (row == int(audio_token_id)).nonzero(as_tuple=False).flatten()
        expected_count = int(expected_counts[row_index].item())
        if audio_positions.numel() > expected_count:
            needs_trim = True
            keep_mask = torch.ones(row.size(0), device=row.device, dtype=torch.bool)
            keep_mask[audio_positions[expected_count:]] = False
            keep_masks.append(keep_mask)
        else:
            keep_masks.append(torch.ones(row.size(0), device=row.device, dtype=torch.bool))
    if not needs_trim:
        return batch

    seq_len = input_ids.size(-1)
    pad_id = 0 if pad_token_id is None else int(pad_token_id)
    trimmed = dict(batch)
    left_pad = True
    sequence_pad_values = {
        "input_ids": pad_id,
        "attention_mask": 0,
        "token_type_ids": 0,
        "position_ids": 0,
        "mm_token_type_ids": 0,
    }
    for key, value in list(batch.items()):
        if (
            key not in sequence_pad_values
            or not isinstance(value, torch.Tensor)
            or value.ndim != 2
            or value.size(0) != input_ids.size(0)
            or value.size(-1) != seq_len
        ):
            continue
        rows = [value[row_index][keep_masks[row_index]] for row_index in range(value.size(0))]
        trimmed[key] = _pad_trimmed_sequence_rows(rows, sequence_pad_values[key], left_pad).contiguous()
    return trimmed


def build_torchao_quant_config() -> TorchAoConfig:
    try:
        from torchao.quantization.quant_api import Int8WeightOnlyConfig

        return TorchAoConfig(Int8WeightOnlyConfig(group_size=128))
    except Exception:
        return TorchAoConfig("int8_weight_only", group_size=128)


def enable_non_reentrant_gradient_checkpointing(model: torch.nn.Module) -> None:
    if not hasattr(model, "gradient_checkpointing_enable"):
        return
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model.gradient_checkpointing_enable()


def should_enable_model_gradient_checkpointing(args: argparse.Namespace) -> bool:
    return bool(args.gradient_checkpointing and not args.use_deepspeed_zero3)


@contextlib.contextmanager
def inference_generate_context(model: torch.nn.Module, args: argparse.Namespace):
    """Context manager that makes model.generate() work correctly during training.

    Two things break KV-cache generation when LoRA+gradient-checkpointing is
    active:

    1. ``apply_lora`` sets ``model.config.use_cache = False`` so that backward
       passes don't allocate the cache.  We restore it to True here.

    2. Gemma4's own ``forward()`` checks ``self.gradient_checkpointing`` on
       every decoder layer and, if True, forces ``past_key_values = None``
       (use_cache=False) regardless of what was requested.  The only way to
       suppress this is to temporarily call ``gradient_checkpointing_disable()``
       before generate and re-enable it after.

    Both items are restored in the ``finally`` block so training continues
    normally after generation.
    """
    needs_gc_toggle = should_enable_model_gradient_checkpointing(args)

    # --- disable gradient checkpointing for the generate call ---
    if needs_gc_toggle and hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    # --- restore use_cache on model config ---
    config = getattr(model, "config", None)
    orig_use_cache = getattr(config, "use_cache", None)
    if config is not None:
        config.use_cache = True

    try:
        yield
    finally:
        # Restore use_cache first
        if config is not None and orig_use_cache is not None:
            config.use_cache = orig_use_cache
        # Re-enable gradient checkpointing
        if needs_gc_toggle:
            enable_non_reentrant_gradient_checkpointing(model)


def build_accelerator(args: argparse.Namespace) -> Accelerator:
    mixed_precision = None if args.mixed_precision == "no" else args.mixed_precision
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        static_graph=bool(args.gradient_checkpointing),
    )
    if not args.use_deepspeed_zero2 and not args.use_deepspeed_zero3:
        return Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            kwargs_handlers=[ddp_kwargs],
        )

    zero3_init_enabled = args.zero3_init_flag or args.use_deepspeed_zero3
    zero_stage = 3 if args.use_deepspeed_zero3 else 2
    hf_ds_config = {
        "train_micro_batch_size_per_gpu": int(args.per_device_train_batch_size),
        "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
        "zero_optimization": {"stage": zero_stage},
    }
    deepspeed_plugin = DeepSpeedPlugin(
        hf_ds_config=hf_ds_config,
        zero_stage=zero_stage,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clipping=args.max_grad_norm if args.max_grad_norm > 0 else None,
        zero3_init_flag=zero3_init_enabled,
        zero3_save_16bit_model=args.zero3_save_16bit_model,
    )
    return Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        deepspeed_plugin=deepspeed_plugin,
        kwargs_handlers=[ddp_kwargs],
    )


def collect_gemma4_zero3_params(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    gathered_params: list[torch.nn.Parameter] = []
    language_model = getattr(module, "language_model", None)
    if language_model is None:
        return gathered_params

    embed_tokens = getattr(language_model, "embed_tokens", None)
    embed_weight = getattr(embed_tokens, "weight", None)
    if isinstance(embed_weight, torch.nn.Parameter):
        gathered_params.append(embed_weight)

    lm_head = getattr(language_model, "lm_head", None)
    lm_head_weight = getattr(lm_head, "weight", None)
    if isinstance(lm_head_weight, torch.nn.Parameter):
        gathered_params.append(lm_head_weight)

    pad_embed = getattr(module, "pad_embed", None)
    pad_embed_weight = getattr(pad_embed, "weight", None)
    if isinstance(pad_embed_weight, torch.nn.Parameter):
        gathered_params.append(pad_embed_weight)

    return gathered_params


def maybe_log_zero3_debug(module: torch.nn.Module, stage: str) -> None:
    if os.environ.get("GEMMA4_ZERO3_DEBUG") != "1":
        return
    language_model = getattr(module, "language_model", None)
    embed_tokens = getattr(language_model, "embed_tokens", None)
    weight = getattr(embed_tokens, "weight", None)
    payload = {
        "event": "gemma4_zero3_debug",
        "stage": stage,
        "module_type": type(module).__name__,
        "language_model_type": type(language_model).__name__ if language_model is not None else None,
        "embed_tokens_type": type(embed_tokens).__name__ if embed_tokens is not None else None,
        "weight_type": type(weight).__name__ if weight is not None else None,
        "weight_dim": int(weight.dim()) if isinstance(weight, torch.Tensor) else None,
        "weight_shape": list(weight.shape) if isinstance(weight, torch.Tensor) else None,
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
    }
    print(json.dumps(payload, ensure_ascii=False))


def infer_zero3_weight_shape(
    target_module: torch.nn.Module,
    original_weight: torch.Tensor,
) -> tuple[int, ...] | None:
    ds_shape = getattr(original_weight, "ds_shape", None)
    if ds_shape is not None:
        target_shape = tuple(int(dim) for dim in ds_shape)
        if len(target_shape) > 1 and original_weight.numel() == math.prod(target_shape):
            return target_shape

    if isinstance(target_module, torch.nn.Embedding):
        target_shape = (int(target_module.num_embeddings), int(target_module.embedding_dim))
        if original_weight.numel() == math.prod(target_shape):
            return target_shape

    if isinstance(target_module, torch.nn.Linear):
        target_shape = (int(target_module.out_features), int(target_module.in_features))
        if original_weight.numel() == math.prod(target_shape):
            return target_shape

    return None


def replace_zero3_weights_for_forward(module: torch.nn.Module) -> list[torch.nn.Module]:
    restored: list[torch.nn.Module] = []
    language_model = getattr(module, "language_model", None)
    candidate_modules = [
        (language_model, "embed_tokens"),
        (language_model, "lm_head"),
        (module, "pad_embed"),
    ]
    for owner_module, attr_name in candidate_modules:
        if owner_module is None:
            continue
        target_module = getattr(owner_module, attr_name, None)
        if target_module is None:
            continue
        original_weight = getattr(target_module, "weight", None)
        if not isinstance(original_weight, torch.Tensor):
            continue
        target_shape = infer_zero3_weight_shape(target_module, original_weight)
        if target_shape is None:
            continue
        reshaped_weight = original_weight.view(*target_shape)
        object.__setattr__(target_module, "weight", reshaped_weight)
        restored.append(target_module)
    return restored


def restore_zero3_module_proxies(
    restored: Iterable[torch.nn.Module],
) -> None:
    for module in restored:
        if "weight" in module.__dict__:
            del module.__dict__["weight"]


@contextlib.contextmanager
def gemma4_zero3_gathered_forward_context(
    model: torch.nn.Module,
    args: argparse.Namespace,
):
    if not args.use_deepspeed_zero3:
        yield
        return

    try:
        from deepspeed import zero
    except ModuleNotFoundError:
        yield
        return

    target_module = None
    for module in model.modules():
        language_model = getattr(module, "language_model", None)
        embed_tokens = getattr(language_model, "embed_tokens", None)
        if getattr(embed_tokens, "weight", None) is not None:
            target_module = module
            break

    if target_module is None:
        yield
        return

    gathered_params = collect_gemma4_zero3_params(target_module)
    if not gathered_params:
        yield
        return

    try:
        context = zero.GatheredParameters(gathered_params, modifier_rank=0, fwd_module=target_module)
    except TypeError:
        context = zero.GatheredParameters(gathered_params, modifier_rank=0)

    with context:
        if not getattr(target_module, "_gemma4_zero3_debug_pre_logged", False):
            maybe_log_zero3_debug(target_module, "pre_replace")
            target_module._gemma4_zero3_debug_pre_logged = True
        restored = replace_zero3_weights_for_forward(target_module)
        try:
            if not getattr(target_module, "_gemma4_zero3_debug_post_logged", False):
                maybe_log_zero3_debug(target_module, "post_replace")
                target_module._gemma4_zero3_debug_post_logged = True
            yield
        finally:
            restore_zero3_module_proxies(restored)


def patch_gemma4_zero3_forward(model: torch.nn.Module, args: argparse.Namespace) -> None:
    if not args.use_deepspeed_zero3:
        return

    try:
        from deepspeed import zero
    except ModuleNotFoundError:
        return

    def maybe_register_external_parameter(owner_module: torch.nn.Module, parameter: torch.nn.Parameter) -> None:
        register_fn = getattr(zero, "register_external_parameter", None)
        if register_fn is None:
            return
        try:
            register_fn(owner_module, parameter)
        except Exception:
            pass

    def gathered_parameters_context(params: list[torch.nn.Parameter], module: torch.nn.Module):
        try:
            return zero.GatheredParameters(params, modifier_rank=0, fwd_module=module)
        except TypeError:
            return zero.GatheredParameters(params, modifier_rank=0)

    for module in model.modules():
        language_model = getattr(module, "language_model", None)
        embed_tokens = getattr(language_model, "embed_tokens", None)
        weight = getattr(embed_tokens, "weight", None)
        if weight is None or getattr(module, "_gemma4_zero3_forward_patched", False):
            continue

        gathered_params = collect_gemma4_zero3_params(module)
        for param in gathered_params:
            maybe_register_external_parameter(module, param)
            if language_model is not None:
                maybe_register_external_parameter(language_model, param)
        module._gemma4_zero3_forward_patched = True


def patch_gemma4_quant_conv_dtype(model: torch.nn.Module, args: argparse.Namespace) -> None:
    if not (args.load_in_4bit or args.load_in_8bit or args.torchao_int8_weight_only):
        return

    for module in model.modules():
        depthwise_conv = getattr(module, "depthwise_conv1d", None)
        weight = getattr(depthwise_conv, "weight", None)
        if depthwise_conv is None or weight is None:
            continue
        if getattr(depthwise_conv, "_gemma4_quant_dtype_patched", False):
            continue

        original_forward = depthwise_conv.forward

        @functools.wraps(original_forward)
        def wrapped_forward(x, *forward_args, __orig_forward=original_forward, __conv=depthwise_conv, **forward_kwargs):
            conv_weight = getattr(__conv, "weight", None)
            target_dtype = getattr(conv_weight, "dtype", None)
            if (
                isinstance(x, torch.Tensor)
                and target_dtype is not None
                and torch.is_floating_point(x)
                and x.dtype != target_dtype
                and getattr(target_dtype, "is_floating_point", False)
            ):
                x = x.to(target_dtype)
            return __orig_forward(x, *forward_args, **forward_kwargs)

        depthwise_conv.forward = wrapped_forward
        depthwise_conv._gemma4_quant_dtype_patched = True


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
            if not (
                module_name == requested_name
                or module_name.endswith(f".{requested_name}")
            ):
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
        raise ValueError(f"None of the requested LoRA target modules were found: {requested}")
    return resolved


def load_model_and_processor(
    args: argparse.Namespace, accelerator: Accelerator | None = None
) -> tuple[Any, torch.nn.Module]:
    if accelerator is not None and torch.cuda.is_available():
        target_device_index = getattr(accelerator, "local_process_index", None)
        if target_device_index is None:
            target_device_index = getattr(accelerator, "process_index", 0)
        torch.cuda.set_device(int(target_device_index))
    try:
        processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)
    except ValueError as exc:
        message = str(exc)
        processor_load_is_gemma4_compat_error = any(
            marker in message
            for marker in (
                "Unrecognized image processor",
                "Unrecognized processing class",
                "Gemma4Processor",
                "model type `gemma4`",
            )
        )
        if not processor_load_is_gemma4_compat_error:
            raise
        if accelerator is None or accelerator.is_local_main_process:
            logging.warning(
                "AutoProcessor could not load the full Gemma4 processor; using "
                "audio/text-only processor fallback. Original error: %s",
                message,
            )
        processor = _AudioTextOnlyGemma4Processor.from_pretrained(args.base_model_path)
    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
        "low_cpu_mem_usage": True,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    if args.torchao_int8_weight_only:
        model_kwargs["quantization_config"] = build_torchao_quant_config()
    if args.load_in_4bit or args.load_in_8bit:
        if accelerator is None:
            raise ValueError(
                "accelerator is required when --load-in-4bit or --load-in-8bit is enabled."
            )
        target_device_index = getattr(accelerator, "local_process_index", None)
        if target_device_index is None:
            target_device_index = getattr(accelerator, "process_index", 0)
        quant_kwargs: dict[str, Any] = {
            "load_in_4bit": args.load_in_4bit,
            "load_in_8bit": args.load_in_8bit,
        }
        if args.load_in_4bit:
            quant_kwargs.update(
                {
                    "bnb_4bit_quant_type": args.bnb_4bit_quant_type,
                    "bnb_4bit_use_double_quant": args.bnb_4bit_use_double_quant,
                    "bnb_4bit_compute_dtype": resolve_explicit_torch_dtype(
                        args.bnb_4bit_compute_dtype
                    ),
                }
            )
        model_kwargs["quantization_config"] = BitsAndBytesConfig(**quant_kwargs)
        # In multi-process training each rank must load the quantized base model
        # onto its own local CUDA device. Using the explicit rank index is more
        # robust than relying on the string form of accelerator.device here.
        model_kwargs["device_map"] = {"": int(target_device_index)}

    if args.use_deepspeed_zero3 and accelerator is not None:
        plugin = accelerator.state.deepspeed_plugin
        zero3_init_enabled = args.zero3_init_flag or args.use_deepspeed_zero3
        with plugin.zero3_init_context_manager(enable=zero3_init_enabled):
            model = AutoModelForImageTextToText.from_pretrained(
                args.base_model_path, trust_remote_code=True, **model_kwargs
            )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.base_model_path, trust_remote_code=True, **model_kwargs
        )
    patch_gemma4_zero3_forward(model, args)
    patch_gemma4_quant_conv_dtype(model, args)
    return processor, model


def get_vocab_size(processor: Any) -> int:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Processor is missing tokenizer, cannot normalize CE loss.")
    if getattr(tokenizer, "vocab_size", None):
        return int(tokenizer.vocab_size)
    return len(tokenizer.get_vocab())


def apply_lora(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, list[str], int, int]:
    try:
        from peft import (
            LoraConfig,
            PeftModel,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "peft is required for LoRA training. Install peft first."
        ) from exc

    effective_gradient_checkpointing = should_enable_model_gradient_checkpointing(args)

    if (
        args.load_in_4bit
        or args.load_in_8bit
        or args.torchao_int8_weight_only
        or effective_gradient_checkpointing
        or args.init_adapter_path is not None
        or args.use_deepspeed_zero3
    ) and hasattr(model, "config"):
        model.config.use_cache = False

    if args.load_in_4bit or args.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,
        )
    if effective_gradient_checkpointing or args.torchao_int8_weight_only:
        enable_non_reentrant_gradient_checkpointing(model)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if args.init_adapter_path is not None:
        peft_model = PeftModel.from_pretrained(
            model,
            str(args.init_adapter_path),
            is_trainable=True,
        )
        if args.kl_coef > 0:
            peft_model.load_adapter(
                str(args.init_adapter_path),
                adapter_name="reference",
                is_trainable=False,
            )
            peft_model.set_adapter(getattr(peft_model, "active_adapter", "default"))
        active_adapter = getattr(peft_model, "active_adapter", "default")
        config = peft_model.peft_config[active_adapter]
        target_modules = sorted(str(module) for module in config.target_modules)
    else:
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
        peft_model = get_peft_model(model, lora_config)

    patch_gemma4_zero3_forward(peft_model, args)
    patch_gemma4_quant_conv_dtype(peft_model, args)
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    return peft_model, target_modules, trainable_params, total_params


def collect_quantization_summary(model: torch.nn.Module) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "is_loaded_in_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
        "is_loaded_in_8bit": bool(getattr(model, "is_loaded_in_8bit", False)),
        "linear4bit_modules": 0,
        "linear8bitlt_modules": 0,
    }
    try:
        import bitsandbytes as bnb

        linear4bit_cls = getattr(bnb.nn, "Linear4bit", None)
        linear8bit_cls = getattr(bnb.nn, "Linear8bitLt", None)
        for module in model.modules():
            if linear4bit_cls is not None and isinstance(module, linear4bit_cls):
                summary["linear4bit_modules"] += 1
            if linear8bit_cls is not None and isinstance(module, linear8bit_cls):
                summary["linear8bitlt_modules"] += 1
    except Exception:
        summary["bitsandbytes_available"] = False
    else:
        summary["bitsandbytes_available"] = True
    return summary


def get_peft_model_for_adapter_switch(model: torch.nn.Module) -> torch.nn.Module:
    current = model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "set_adapter") and hasattr(current, "active_adapter"):
            return current
        for attr_name in ("module", "model"):
            child = getattr(current, attr_name, None)
            if child is not None and child is not current:
                current = child
                break
        else:
            break
    return model


@contextlib.contextmanager
def peft_adapter_context(model: torch.nn.Module, adapter_name: str | None):
    if adapter_name is None:
        yield
        return
    peft_model = get_peft_model_for_adapter_switch(model)
    if not hasattr(peft_model, "set_adapter"):
        yield
        return
    previous_adapter = getattr(peft_model, "active_adapter", None)
    try:
        peft_model.set_adapter(adapter_name)
        yield
    finally:
        if previous_adapter is not None:
            peft_model.set_adapter(previous_adapter)


def get_pad_token_id(processor: Any) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return tokenizer.eos_token_id


def get_generation_stop_token_ids(processor: Any) -> int | list[int] | None:
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


def normalize_ce_loss(raw_ce_loss: torch.Tensor, vocab_size: int) -> torch.Tensor:
    scale = math.log(max(vocab_size, 2))
    normalized = raw_ce_loss.float() / max(scale, 1e-6)
    return normalized.clamp(min=0.0, max=1.0)


def zscore_normalize_rewards(reward_tensor: torch.Tensor) -> torch.Tensor:
    if reward_tensor.numel() <= 1:
        return torch.zeros_like(reward_tensor, dtype=torch.float32)
    reward_tensor = reward_tensor.float()
    reward_mean = reward_tensor.mean()
    reward_std = reward_tensor.std(unbiased=False)
    return (reward_tensor - reward_mean) / (reward_std + 1e-6)


def score_assistant_texts_with_teacher_forcing(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    assistant_texts: list[str],
    accelerator: Accelerator,
    args: argparse.Namespace,
    adapter_name: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    prompt_inputs = ensure_batch_dims(prompt_inputs)
    audio_token_id = get_audio_token_id(processor)
    pad_token_id = get_pad_token_id(processor)
    prompt_inputs = align_audio_feature_mask(prompt_inputs)
    prompt_inputs = trim_surplus_audio_tokens(prompt_inputs, audio_token_id, pad_token_id)
    prompt_length = prompt_inputs["input_ids"].size(-1)

    full_conversations = [
        prompt_messages
        + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": assistant_text}],
            }
        ]
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
    full_inputs = ensure_batch_dims(full_inputs)
    full_inputs = move_batch_to_device(
        full_inputs,
        accelerator.device,
        float_dtype=get_runtime_float_dtype(args),
        audio_token_id=audio_token_id,
        pad_token_id=pad_token_id,
    )

    with peft_adapter_context(model, adapter_name):
        with gemma4_zero3_gathered_forward_context(model, args):
            outputs = model(**full_inputs)
    logits = outputs.logits[:, :-1, :]
    labels = full_inputs["input_ids"][:, 1:]
    target_logits = logits.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    normalizers = torch.logsumexp(logits, dim=-1)
    token_log_probs = target_logits.float() - normalizers.float()
    token_log_probs = torch.nan_to_num(token_log_probs, nan=-1e4, posinf=0.0, neginf=-1e4)

    seq_positions = torch.arange(labels.shape[1], device=labels.device).unsqueeze(0)
    target_mask = full_inputs["attention_mask"][:, 1:].bool() & (
        seq_positions >= max(prompt_length - 1, 0)
    )
    token_counts = target_mask.sum(dim=1).clamp_min(1)
    avg_log_probs = (token_log_probs * target_mask).sum(dim=1) / token_counts
    avg_log_probs = torch.nan_to_num(avg_log_probs, nan=-1e4, posinf=0.0, neginf=-1e4)
    avg_nll = -avg_log_probs
    return avg_log_probs, avg_nll


def generate_candidate_texts(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    accelerator: Accelerator,
    args: argparse.Namespace,
) -> list[str]:
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
    prompt_inputs = ensure_batch_dims(prompt_inputs)
    prompt_inputs = move_batch_to_device(
        prompt_inputs,
        accelerator.device,
        float_dtype=get_runtime_float_dtype(args),
        audio_token_id=get_audio_token_id(processor),
        pad_token_id=get_pad_token_id(processor),
    )
    prompt_length = prompt_inputs["input_ids"].size(-1)
    pad_token_id = get_pad_token_id(processor)
    eos_token_id = get_generation_stop_token_ids(processor)

    # Generation strategy switch:
    #   * "sample"      — independent multinomial samples. Fragile on low-entropy
    #                     SFT models (multilingual leakage, early-EOS).
    #   * "beam"        — plain beam search (top-K best hypotheses). High quality
    #                     but candidates may be near-duplicates.
    #   * "beam_sample" — beam search + per-step multinomial sampling. Stays on
    #                     high-probability manifold (no leakage / no early-EOS)
    #                     and gets diversity from sampling. Recommended default.
    #
    # Note: diverse beam search (num_beam_groups > 1) was removed — newer
    # transformers (>=4.55) require trust_remote_code + a custom_generate repo.
    strategy = getattr(args, "generation_strategy", "beam_sample")

    # Manual logits processors (Gemma4's GenerationConfig silently drops
    # top_p / top_k kwargs — see "not valid and may be ignored" warning).
    from transformers import (
        LogitsProcessorList,
        NoRepeatNGramLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopKLogitsWarper,
        TopPLogitsWarper,
    )

    processors = LogitsProcessorList()
    if args.repetition_penalty and args.repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(args.repetition_penalty))
    if args.no_repeat_ngram_size and args.no_repeat_ngram_size > 0:
        processors.append(NoRepeatNGramLogitsProcessor(args.no_repeat_ngram_size))
    if strategy != "beam":
        # Beam (deterministic) ignores temperature/top_k/top_p; only apply for
        # sample / beam_sample paths.
        if args.temperature and args.temperature != 1.0:
            processors.append(TemperatureLogitsWarper(args.temperature))
        if args.top_k and args.top_k > 0:
            processors.append(TopKLogitsWarper(args.top_k))
        if args.top_p and 0.0 < args.top_p < 1.0:
            processors.append(TopPLogitsWarper(args.top_p))

    # use_cache=True is needed even though model.config.use_cache was set to
    # False by apply_lora / gradient-checkpointing setup.  Passing it
    # explicitly here overrides the config for the generate call so that the
    # KV cache is used.  Without this, Gemma4 re-computes all past hidden
    # states at every autoregressive step, which breaks the positional
    # encoding logic and causes degenerate repetition loops ("尹尹尹尹…").
    common_kwargs: dict[str, Any] = dict(
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_candidates,
        logits_processor=processors,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        use_cache=True,
    )

    if strategy == "beam_sample":
        gen_kwargs = dict(
            do_sample=True,
            num_beams=args.num_candidates,
            early_stopping=True,
            length_penalty=float(getattr(args, "length_penalty", 1.0)),
            **common_kwargs,
        )
    elif strategy == "beam":
        gen_kwargs = dict(
            do_sample=False,
            num_beams=max(args.num_candidates, 2 * args.num_candidates),
            early_stopping=True,
            length_penalty=float(getattr(args, "length_penalty", 1.0)),
            **common_kwargs,
        )
    else:  # "sample"
        gen_kwargs = dict(
            do_sample=True,
            **common_kwargs,
        )

    _unwrapped_for_gen = accelerator.unwrap_model(model)

    with inference_generate_context(_unwrapped_for_gen, args):
        with torch.inference_mode():
            with gemma4_zero3_gathered_forward_context(_unwrapped_for_gen, args):
                generated = _unwrapped_for_gen.generate(
                    **prompt_inputs,
                    **gen_kwargs,
                )

    generated = ensure_tensor_has_batch_dim(generated)
    generated_only = generated[:, prompt_length:]
    texts = processor.batch_decode(
        generated_only,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    # Diagnostic: when DEBUG_GREEDY_PROBE=1, also run a pure greedy generation
    # on the same audio prompt and print it side-by-side. If the greedy probe
    # produces sensible Chinese while `texts` are degenerate, the bug is in
    # the multi-candidate path; if both are degenerate, the model itself
    # cannot translate this training sample.
    if os.environ.get("DEBUG_GREEDY_PROBE") == "1":
        try:
            with inference_generate_context(_unwrapped_for_gen, args):
                with torch.inference_mode():
                    with gemma4_zero3_gathered_forward_context(
                        _unwrapped_for_gen, args
                    ):
                        greedy_out = _unwrapped_for_gen.generate(
                            **prompt_inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            num_beams=1,
                            num_return_sequences=1,
                            pad_token_id=pad_token_id,
                            eos_token_id=eos_token_id,
                            use_cache=True,
                        )
            greedy_out = ensure_tensor_has_batch_dim(greedy_out)
            greedy_text = processor.batch_decode(
                greedy_out[:, prompt_length:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            probe_payload = {
                "_debug": "greedy_probe",
                "id": sample.get("id"),
                "reference": sample.get("reference", ""),
                "greedy": normalize_text(greedy_text),
                "candidates": [normalize_text(t) for t in texts],
                "prompt_length": int(prompt_length),
                "strategy": strategy,
            }
            print(json.dumps(probe_payload, ensure_ascii=False), flush=True)
        except Exception as exc:  # pragma: no cover
            print(
                json.dumps(
                    {"_debug": "greedy_probe_failed", "error": str(exc)},
                    ensure_ascii=False,
                ),
                flush=True,
            )

    return [normalize_text(text) for text in texts]


def compute_sentence_scores(prediction: str, reference: str) -> tuple[float, float]:
    sacrebleu = get_sacrebleu()
    bleu = (
        sacrebleu.sentence_bleu(
            prediction,
            [reference],
            tokenize="zh",
            use_effective_order=True,
        ).score
        / 100.0
    )
    chrf = sacrebleu.sentence_chrf(
        prediction,
        [reference],
        word_order=0,
    ).score / 100.0
    return bleu, chrf


def _longest_common_substring_len(s: str, t: str) -> int:
    """Length of the longest contiguous substring common to s and t. O(|s|·|t|) time."""
    if not s or not t:
        return 0
    n = len(t)
    best = 0
    prev = [0] * (n + 1)
    for ch in s:
        curr = [0] * (n + 1)
        for j in range(n):
            if ch == t[j]:
                curr[j + 1] = prev[j] + 1
                if curr[j + 1] > best:
                    best = curr[j + 1]
        prev = curr
    return best


def compute_key_recall(
    prediction: str,
    gold_keys: list[str],
    match_mode: str = "normalized_exact",
) -> float:
    if not gold_keys:
        return 1.0
    if match_mode == "fuzzy_substring":
        # Partial credit: longest common substring / |key|.
        # "莫斯科市议会" vs pred containing "莫斯科" → 3/6 = 0.5
        norm_pred = normalize_key_text(prediction)
        total = 0.0
        valid = 0
        for key in gold_keys:
            norm_key = normalize_key_text(str(key))
            if not norm_key:
                continue
            lcs = _longest_common_substring_len(norm_key, norm_pred)
            total += lcs / len(norm_key)
            valid += 1
        return total / valid if valid else 1.0
    if match_mode == "normalized_exact":
        match_prediction = normalize_key_text(prediction)
        match_keys = [normalize_key_text(str(key)) for key in gold_keys]
    else:
        match_prediction = prediction
        match_keys = [str(key) for key in gold_keys]
    matches = sum(1 for key in match_keys if key and key in match_prediction)
    return matches / len(gold_keys)


_ENTITY_SKIP_LABELS: frozenset[str] = frozenset({"PER"})


def _fuzzy_entity_recall(
    pred_entities: list[Any],
    ref_entities: list[Any],
    skip_labels: frozenset[str] = _ENTITY_SKIP_LABELS,
) -> float:
    """Recall-oriented fuzzy match: for every ref entity, find the best LCS
    ratio among pred entities.  Score = mean of per-ref best scores.
    Labels in skip_labels are excluded (default: PER, because Ky→Zh person
    names are phonetic transliterations that don't carry semantic content)."""
    active_refs = [r for r in ref_entities if str(getattr(r, "label", "")).upper() not in skip_labels]
    if not active_refs:
        return 1.0
    active_preds = [p for p in pred_entities if str(getattr(p, "label", "")).upper() not in skip_labels]
    if not active_preds:
        return 0.0
    total = 0.0
    for ref in active_refs:
        ref_text = normalize_key_text(str(ref.text))
        if not ref_text:
            continue
        best = max(
            (
                _longest_common_substring_len(ref_text, normalize_key_text(str(p.text))) / len(ref_text)
                for p in active_preds
                if normalize_key_text(str(p.text))
            ),
            default=0.0,
        )
        total += best
    return total / len(active_refs)


def _micro_f1(tp: int, pred_total: int, ref_total: int) -> float:
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / ref_total if ref_total else 0.0
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


class EntityRewardComputer:
    def __init__(self, args: argparse.Namespace, weight_config: dict[str, Any]) -> None:
        self.mode = str(weight_config.get("entity_reward_mode", "key_recall"))
        self.tau = float(weight_config.get("entity_soft_tau", 0.6))
        self.extractor = None
        self.gemma_extractor = None
        self.embedder = None
        self.entity_cache: dict[str, list[Any]] = {}
        self.embedding_cache: dict[str, torch.Tensor] = {}
        self.args = args
        # PER is included in the reward by default. Pass --no-entity-include-per
        # to exclude it (Ky→Zh person names are phonetic transliterations).
        self.skip_labels: frozenset[str] = (
            frozenset() if getattr(args, "entity_include_per", True)
            else _ENTITY_SKIP_LABELS
        )
        # Optional: a model+tokenizer shared with the training loop, set after
        # the model is loaded via set_shared_model().  Used by
        # `entity_gemma_fuzzy` mode so we don't allocate a second copy in VRAM.
        self._shared_model: Any = None
        self._shared_tokenizer: Any = None

    def set_shared_model(self, model: Any, processor: Any) -> None:
        """Register the training model so entity_gemma_fuzzy reuses it.

        Called once after model+processor are loaded in train().  The tokenizer
        is taken from processor.tokenizer (Gemma4Processor exposes it that way).
        """
        self._shared_model = model
        self._shared_tokenizer = getattr(processor, "tokenizer", processor)
        # If a standalone extractor was already created, drop it so the next
        # extract() call picks up the shared one.
        self.gemma_extractor = None

    def _load_entity_backend(self) -> tuple[Any, Any, Any, Any, Any]:
        from eval_suite.entity_backend import (
            HanLPEntityExtractor,
            MBertEmbedder,
            cosine_similarity_matrix,
            soft_entity_matching,
        )
        from eval_suite.types import EmbeddingConfig, Entity, NERConfig

        return (
            HanLPEntityExtractor,
            MBertEmbedder,
            cosine_similarity_matrix,
            soft_entity_matching,
            (EmbeddingConfig, Entity, NERConfig),
        )

    def _ensure_extractor(self) -> Any:
        if self.extractor is None:
            HanLPEntityExtractor, _, _, _, configs = self._load_entity_backend()
            _, _, NERConfig = configs
            extractor_device = getattr(self.args, "entity_extractor_device", None) or "cpu"
            ner_config = NERConfig(
                tokenizer_model=self.args.ner_tokenizer_model,
                ner_model=self.args.ner_model,
                tokenizer_path=self.args.ner_tokenizer_path,
                ner_path=self.args.ner_path,
            )
            try:
                self.extractor = HanLPEntityExtractor(ner_config, device=extractor_device)
            except TypeError as exc:
                if "device" not in str(exc):
                    raise
                self.extractor = HanLPEntityExtractor(ner_config)
        return self.extractor

    def _ensure_gemma_extractor(self) -> Any:
        if self.gemma_extractor is None:
            from eval_suite.entity_backend import GemmaEntityExtractor
            if self._shared_model is not None and self._shared_tokenizer is not None:
                # Reuse the training model — no extra VRAM allocation.
                self.gemma_extractor = GemmaEntityExtractor.from_shared_model(
                    self._shared_model, self._shared_tokenizer
                )
            else:
                model_path = getattr(self.args, "gemma_ner_model_path", None) or "gemma-4-E2B-it"
                extractor_device = getattr(self.args, "entity_extractor_device", None) or "cpu"
                self.gemma_extractor = GemmaEntityExtractor(model_path, device=extractor_device)
        return self.gemma_extractor

    def _ensure_embedder(self) -> Any:
        if self.embedder is None:
            _, MBertEmbedder, _, _, configs = self._load_entity_backend()
            EmbeddingConfig, _, _ = configs
            embedder_device = getattr(self.args, "entity_embedding_device", None) or "cpu"
            self.embedder = MBertEmbedder(
                EmbeddingConfig(
                    model_name=self.args.entity_embedding_model,
                    max_length=self.args.entity_embedding_max_length,
                    pooling=self.args.entity_embedding_pooling,
                ),
                device=embedder_device,
            )
        return self.embedder

    def _entity_objects(self, rows: list[dict[str, str]]) -> list[Any]:
        _, _, _, _, configs = self._load_entity_backend()
        _, Entity, _ = configs
        entities = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            text = str(row.get("text", "") or "").strip()
            label = str(row.get("label", "") or "MISC").strip().upper()
            if not text:
                continue
            key = (text, label)
            if key in seen:
                continue
            seen.add(key)
            entities.append(Entity(text=text, label=label))
        return entities

    def reference_entities(self, sample: dict[str, Any]) -> list[Any]:
        gold_entities = sample.get("gold_entities", [])
        if gold_entities:
            return self._entity_objects(gold_entities)
        reference = str(sample.get("reference", "") or "")
        cache_key = f"ref::{reference}"
        if cache_key not in self.entity_cache:
            self.entity_cache[cache_key] = self._ensure_extractor().extract(reference)
        return self.entity_cache[cache_key]

    def prediction_entities(self, prediction: str) -> list[Any]:
        cache_key = f"pred::{prediction}"
        if cache_key not in self.entity_cache:
            self.entity_cache[cache_key] = self._ensure_extractor().extract(prediction)
        return self.entity_cache[cache_key]

    def _encode_entities(self, entities: list[Any]) -> torch.Tensor:
        texts = [str(entity.text) for entity in entities]
        missing = [text for text in texts if text not in self.embedding_cache]
        if missing:
            embeddings = self._ensure_embedder().encode(missing)
            for text, embedding in zip(missing, embeddings):
                self.embedding_cache[text] = embedding.cpu()
        if not texts:
            return torch.empty((0, 1), dtype=torch.float32)
        return torch.stack([self.embedding_cache[text] for text in texts], dim=0)

    # ── debug counter: print entity details for the first N score() calls ──
    _entity_debug_remaining: int = int(os.environ.get("DEBUG_ENTITY_SCORE", "0"))

    def score(self, prediction: str, sample: dict[str, Any]) -> tuple[float, dict[str, float]]:
        if self.mode == "key_recall":
            score = compute_key_recall(
                prediction,
                sample.get("gold_keys", []),
                match_mode=self.args.key_match_mode,
            )
            return score, {"entity_score": score, "entity_key_recall": score}

        if self.mode == "entity_substring":
            # Direct LCS substring match of gold_entities against prediction text.
            # No Gemma inference needed on the prediction side — same logic as
            # key_recall but using the richer gold_entities sidecar instead of gold_keys.
            ref_entities = self.reference_entities(sample)
            active_refs = [r for r in ref_entities
                           if str(getattr(r, "label", "")).upper() not in self.skip_labels]
            if not active_refs:
                return 1.0, {"entity_score": 1.0, "entity_substring": 1.0}
            norm_pred = normalize_key_text(prediction)
            total = 0.0
            for ref in active_refs:
                ref_text = normalize_key_text(str(ref.text))
                if not ref_text:
                    continue
                lcs = _longest_common_substring_len(ref_text, norm_pred)
                total += lcs / len(ref_text)
            score = total / len(active_refs)
            return score, {"entity_score": score, "entity_substring": score}

        ref_entities = self.reference_entities(sample)
        if not ref_entities:
            return 1.0, {"entity_score": 1.0, "entity_em": 1.0, "entity_soft": 1.0}

        if self.mode == "entity_gemma_fuzzy":
            # Use Gemma to extract entities from the prediction (same model/prompt as the
            # ref sidecar), then do recall-oriented LCS matching against ref entities.
            # NOTE: we skip prediction_entities() here to avoid loading HanLP unnecessarily.
            cache_key = f"gemma_pred::{prediction}"
            if cache_key not in self.entity_cache:
                self.entity_cache[cache_key] = self._ensure_gemma_extractor().extract(prediction)
            gemma_pred_entities = self.entity_cache[cache_key]
            if EntityRewardComputer._entity_debug_remaining > 0:
                EntityRewardComputer._entity_debug_remaining -= 1
                import json as _json
                print(
                    "[ENTITY_DEBUG] " + _json.dumps({
                        "mode": self.mode,
                        "prediction": prediction[:120],
                        "pred_entities": [{"text": e.text, "label": e.label} for e in gemma_pred_entities],
                        "ref_entities":  [{"text": e.text, "label": e.label} for e in ref_entities],
                    }, ensure_ascii=False),
                    flush=True,
                )
            score = _fuzzy_entity_recall(gemma_pred_entities, ref_entities, self.skip_labels)
            return score, {"entity_score": score, "entity_gemma_fuzzy": score}

        pred_entities = self.prediction_entities(prediction)

        if EntityRewardComputer._entity_debug_remaining > 0:
            EntityRewardComputer._entity_debug_remaining -= 1
            import json as _json
            print(
                "[ENTITY_DEBUG] " + _json.dumps({
                    "mode": self.mode,
                    "tau": self.tau,
                    "prediction": prediction[:120],
                    "pred_entities": [{"text": e.text, "label": e.label} for e in pred_entities],
                    "ref_entities":  [{"text": e.text, "label": e.label} for e in ref_entities],
                }, ensure_ascii=False),
                flush=True,
            )

        if self.mode == "entity_em":
            matched_ref = [False] * len(ref_entities)
            tp = 0
            for entity in pred_entities:
                for index, ref_entity in enumerate(ref_entities):
                    if matched_ref[index]:
                        continue
                    if entity.label == ref_entity.label and entity.text == ref_entity.text:
                        matched_ref[index] = True
                        tp += 1
                        break
            score = _micro_f1(tp, len(pred_entities), len(ref_entities))
            return score, {"entity_score": score, "entity_em": score}

        if self.mode == "entity_soft":
            _, _, cosine_similarity_matrix, soft_entity_matching, _ = self._load_entity_backend()
            pred_embeddings = self._encode_entities(pred_entities)
            ref_embeddings = self._encode_entities(ref_entities)
            similarity_matrix = cosine_similarity_matrix(pred_embeddings, ref_embeddings)
            matches = soft_entity_matching(pred_entities, ref_entities, similarity_matrix, self.tau)
            score = _micro_f1(len(matches), len(pred_entities), len(ref_entities))
            return score, {"entity_score": score, "entity_soft": score}

        return 0.0, {"entity_score": 0.0}


def build_entity_rewarder(
    args: argparse.Namespace,
    weight_config: dict[str, Any],
) -> EntityRewardComputer | None:
    if weight_config["reward_weights"].get("key", 0.0) <= 0:
        return None
    if weight_config.get("entity_reward_mode") == "none":
        return None
    return EntityRewardComputer(args, weight_config)


def compute_len_ratio(prediction: str, reference: str) -> float:
    pred_len = max(len(normalize_key_text(prediction)), 1)
    ref_len = max(len(normalize_key_text(reference)), 1)
    return pred_len / ref_len


def compute_reference_ce_loss(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    accelerator: Accelerator,
    args: argparse.Namespace,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, avg_nll = score_assistant_texts_with_teacher_forcing(
        model=model,
        processor=processor,
        sample=sample,
        assistant_texts=[sample["reference"]],
        accelerator=accelerator,
        args=args,
    )
    raw_ce = avg_nll[0]
    return raw_ce, normalize_ce_loss(raw_ce, vocab_size)


_DEBUG_REWARD_DUMP_REMAINING: list[int] = []


def _debug_reward_dump_init() -> None:
    if not _DEBUG_REWARD_DUMP_REMAINING:
        try:
            limit = int(os.environ.get("DEBUG_REWARD_DUMP", "0"))
        except ValueError:
            limit = 0
        _DEBUG_REWARD_DUMP_REMAINING.append(max(limit, 0))


def _parse_debug_step_ranges(raw_value: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for part in raw_value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
        else:
            start = end = int(part)
        if end < start:
            start, end = end, start
        ranges.append((start, end))
    return ranges


def _debug_reward_context_enabled(context: dict[str, Any] | None) -> bool:
    if not context:
        return False
    raw_ranges = os.environ.get("DEBUG_REWARD_DUMP_STEPS", "").strip()
    if not raw_ranges:
        return False
    try:
        step = int(context.get("update_step", 0))
        ranges = _parse_debug_step_ranges(raw_ranges)
    except ValueError:
        return False
    return any(start <= step <= end for start, end in ranges)


def compute_candidate_reward(
    prediction: str,
    sample: dict[str, Any],
    weight_config: dict[str, Any],
    candidate_index: int | None = None,
) -> tuple[float, dict[str, float]]:
    bleu, chrf = compute_sentence_scores(prediction, sample["reference"])
    key_recall_substring = compute_key_recall(
        prediction,
        sample.get("gold_keys", []),
        match_mode=weight_config["key_match_mode"],
    )
    entity_score = 0.0
    entity_metrics: dict[str, float] = {}
    entity_rewarder = weight_config.get("_entity_rewarder")
    if weight_config["reward_weights"]["key"] > 0:
        if entity_rewarder is not None:
            entity_score, entity_metrics = entity_rewarder.score(prediction, sample)
        else:
            entity_score = key_recall_substring
            entity_metrics = {"entity_score": entity_score, "entity_key_recall": entity_score}
    reward = (
        weight_config["reward_weights"]["bleu"] * bleu
        + weight_config["reward_weights"]["chrf"] * chrf
        + weight_config["reward_weights"]["key"] * entity_score
    )
    _debug_reward_dump_init()
    debug_context = weight_config.get("_debug_context")
    should_dump = False
    if _DEBUG_REWARD_DUMP_REMAINING and _DEBUG_REWARD_DUMP_REMAINING[0] > 0:
        _DEBUG_REWARD_DUMP_REMAINING[0] -= 1
        should_dump = True
    if _debug_reward_context_enabled(debug_context):
        should_dump = True
    if should_dump:
        gold_keys = sample.get("gold_keys", [])
        norm_pred = normalize_key_text(prediction)
        norm_keys = [normalize_key_text(str(k)) for k in gold_keys]
        _match_mode = weight_config["key_match_mode"]
        if _match_mode == "fuzzy_substring":
            key_debug = [
                {"key": k, "lcs": _longest_common_substring_len(k, norm_pred),
                 "score": round(_longest_common_substring_len(k, norm_pred) / len(k), 4) if k else 0}
                for k in norm_keys if k
            ]
        else:
            key_debug = [{"key": k, "hit": k in norm_pred} for k in norm_keys if k]
        print(json.dumps({
            "_debug": "reward_dump",
            "update_step": debug_context.get("update_step") if debug_context else None,
            "global_step_before_update": debug_context.get("global_step") if debug_context else None,
            "epoch": debug_context.get("epoch") if debug_context else None,
            "batch_index": debug_context.get("batch_index") if debug_context else None,
            "sample_index_in_batch": debug_context.get("sample_index_in_batch") if debug_context else None,
            "candidate_index": candidate_index,
            "source_index": sample.get("source_index"),
            "id": sample.get("id"),
            "audio_path": sample.get("audio_path"),
            "reference": sample.get("reference", "")[:120],
            "prediction": prediction[:120],
            "norm_pred": norm_pred[:120],
            "gold_keys": gold_keys,
            "gold_entities": sample.get("gold_entities", []),
            "key_match_mode": _match_mode,
            "entity_reward_mode": weight_config.get("entity_reward_mode", "key_recall"),
            "key_details": key_debug,
            "key_recall": key_recall_substring,
            "entity_score": entity_score,
            **{k: v for k, v in entity_metrics.items() if k != "entity_score"},
            "bleu": bleu,
            "chrf": chrf,
        }, ensure_ascii=False), flush=True)
    return reward, {
        "bleu": bleu,
        "chrf": chrf,
        "key_recall": key_recall_substring,
        "entity_score": entity_score,
        **entity_metrics,
    }


def compute_group_risk_loss_for_sample(
    model: torch.nn.Module,
    processor: Any,
    sample: dict[str, Any],
    accelerator: Accelerator,
    args: argparse.Namespace,
    weight_config: dict[str, Any],
    vocab_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    ce_loss_norm = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    if weight_config["ce_weight"] > 0:
        _, ce_loss_norm = compute_reference_ce_loss(
            model=model,
            processor=processor,
            sample=sample,
            accelerator=accelerator,
            args=args,
            vocab_size=vocab_size,
        )

    risk_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)
    reward_stats = {
        "mean_reward": 0.0,
        "best_minus_mean_reward": 0.0,
        "mean_reward_zscore": 0.0,
        "reward_zscore_std": 0.0,
        "mean_bleu": 0.0,
        "mean_chrf": 0.0,
        "mean_key_recall": 0.0,
        "mean_entity_score": 0.0,
    }
    if weight_config["risk_objective_weight"] > 0:
        candidates = generate_candidate_texts(model, processor, sample, accelerator, args)
        avg_log_probs, _ = score_assistant_texts_with_teacher_forcing(
            model=model,
            processor=processor,
            sample=sample,
            assistant_texts=candidates,
            accelerator=accelerator,
            args=args,
        )
        kl_loss = torch.zeros((), device=avg_log_probs.device, dtype=torch.float32)
        if args.kl_coef > 0:
            with torch.no_grad():
                reference_log_probs, _ = score_assistant_texts_with_teacher_forcing(
                    model=model,
                    processor=processor,
                    sample=sample,
                    assistant_texts=candidates,
                    accelerator=accelerator,
                    args=args,
                    adapter_name="reference",
                )
            policy_log_dist = torch.log_softmax(args.group_policy_scale * avg_log_probs.float(), dim=0)
            reference_log_dist = torch.log_softmax(
                args.group_policy_scale * reference_log_probs.float(), dim=0
            )
            policy_dist = policy_log_dist.exp()
            kl_loss = (policy_dist * (policy_log_dist - reference_log_dist)).sum()
            kl_loss = torch.nan_to_num(kl_loss, nan=0.0, posinf=1.0, neginf=0.0)

        rewards: list[float] = []
        bleu_scores: list[float] = []
        chrf_scores: list[float] = []
        key_scores: list[float] = []
        entity_scores: list[float] = []
        for candidate_index, candidate in enumerate(candidates):
            reward, score_row = compute_candidate_reward(
                candidate,
                sample,
                weight_config,
                candidate_index=candidate_index,
            )
            rewards.append(reward)
            bleu_scores.append(score_row["bleu"])
            chrf_scores.append(score_row["chrf"])
            key_scores.append(score_row["key_recall"])
            entity_scores.append(score_row.get("entity_score", score_row["key_recall"]))

        reward_tensor = torch.tensor(rewards, device=avg_log_probs.device, dtype=torch.float32)
        reward_zscore = zscore_normalize_rewards(reward_tensor)
        # After z-score normalization, minimizing negative reward is equivalent to
        # maximizing the standardized reward under the group objective.
        risk_tensor = -reward_zscore
        stabilized_log_probs = torch.nan_to_num(
            args.group_policy_scale * avg_log_probs.float(),
            nan=-1e4,
            posinf=0.0,
            neginf=-1e4,
        )
        candidate_probs = torch.softmax(stabilized_log_probs, dim=0)
        candidate_probs = torch.nan_to_num(candidate_probs, nan=0.0, posinf=1.0, neginf=0.0)
        risk_loss = (candidate_probs * risk_tensor).sum()
        risk_loss = torch.nan_to_num(risk_loss, nan=1.0, posinf=1.0, neginf=1.0)
        reward_stats = {
            "mean_reward": float(sum(rewards) / len(rewards)),
            "best_minus_mean_reward": float(
                reward_tensor.max().detach().cpu().item() - reward_tensor.mean().detach().cpu().item()
            ),
            "mean_reward_zscore": float(reward_zscore.mean().detach().cpu().item()),
            "reward_zscore_std": float(reward_zscore.std(unbiased=False).detach().cpu().item()),
            "mean_bleu": float(sum(bleu_scores) / len(bleu_scores) * 100.0),
            "mean_chrf": float(sum(chrf_scores) / len(chrf_scores) * 100.0),
            "mean_key_recall": float(sum(key_scores) / len(key_scores)),
            "mean_entity_score": float(sum(entity_scores) / len(entity_scores)),
        }

    else:
        kl_loss = torch.zeros((), device=accelerator.device, dtype=torch.float32)

    total_loss = (
        weight_config["risk_objective_weight"] * risk_loss
        + weight_config["ce_weight"] * ce_loss_norm
        + args.kl_coef * kl_loss
    )
    total_loss = torch.nan_to_num(total_loss, nan=1.0, posinf=1.0, neginf=1.0)
    metrics = {
        "loss": float(total_loss.detach().cpu().item()),
        "policy_loss": float(risk_loss.detach().cpu().item()),
        "objective_loss": float(risk_loss.detach().cpu().item()),
        "risk_loss": float(risk_loss.detach().cpu().item()),
        "grpo_objective": getattr(args, "grpo_objective", "group_relative_risk_kl"),
        "kl_loss": float(kl_loss.detach().cpu().item()),
        "kl_coef": float(args.kl_coef),
        "ce_loss": float(ce_loss_norm.detach().cpu().item()),
        **reward_stats,
    }
    return total_loss, metrics


def generate_predictions(
    model: torch.nn.Module,
    processor: Any,
    samples: list[dict[str, Any]],
    accelerator: Accelerator,
    batch_size: int,
    max_new_tokens: int,
    sampling_rate: int,
    args: argparse.Namespace,
) -> tuple[list[str], int]:
    predictions: list[str] = []
    skipped_audio_errors = 0
    pad_token_id = get_pad_token_id(processor)
    eos_token_id = get_generation_stop_token_ids(processor)
    total_batches = math.ceil(len(samples) / max(batch_size, 1))
    progress_bar = tqdm(
        iter_batches(samples, batch_size),
        total=total_batches,
        desc="Validation",
        disable=not accelerator.is_local_main_process,
    )

    def _run_generate_for_messages(prompt_messages_batch: list[list[dict[str, Any]]]) -> list[str]:
        model_inputs = processor.apply_chat_template(
            prompt_messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            processor_kwargs={"sampling_rate": sampling_rate},
        )
        model_inputs = ensure_batch_dims(model_inputs)
        model_inputs = move_batch_to_device(
            model_inputs,
            accelerator.device,
            float_dtype=get_runtime_float_dtype(args),
            audio_token_id=get_audio_token_id(processor),
            pad_token_id=get_pad_token_id(processor),
        )

        _val_unwrapped = accelerator.unwrap_model(model)
        with inference_generate_context(_val_unwrapped, args):
            with torch.inference_mode():
                with gemma4_zero3_gathered_forward_context(_val_unwrapped, args):
                    generated = _val_unwrapped.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        use_cache=True,
                    )

        prompt_length = model_inputs["input_ids"].size(-1)
        generated = ensure_tensor_has_batch_dim(generated)
        generated_only = generated[:, prompt_length:]
        decoded = processor.batch_decode(
            generated_only,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [normalize_text(text) for text in decoded]

    for batch in progress_bar:
        prompt_messages = [build_prompt_messages(sample, sampling_rate) for sample in batch]
        try:
            batch_predictions = _run_generate_for_messages(prompt_messages)
            predictions.extend(batch_predictions)
        except Exception as exc:
            # Fall back to per-sample generation so validation can continue even when
            # one sample has malformed or incompatible audio inputs.
            if accelerator.is_local_main_process:
                tqdm.write(
                    "Validation batch preprocessing failed; "
                    "falling back to per-sample generation for this batch."
                )
            for sample, single_prompt in zip(batch, prompt_messages):
                try:
                    single_prediction = _run_generate_for_messages([single_prompt])
                except Exception as single_exc:
                    sample_id = sample.get("id", "<unknown>")
                    audio_path = sample.get("audio_path", "<unknown>")
                    skipped_audio_errors += 1
                    if accelerator.is_local_main_process:
                        tqdm.write(
                            "Validation sample skipped due to audio preprocessing error: "
                            f"id={sample_id}, audio_path={audio_path}, error={single_exc}"
                        )
                    # Keep alignment with references for corpus metrics.
                    single_prediction = [""]
                predictions.extend(single_prediction)

    progress_bar.close()
    return predictions, skipped_audio_errors


def evaluate_validation(
    model: torch.nn.Module,
    processor: Any,
    samples: list[dict[str, Any]],
    accelerator: Accelerator,
    args: argparse.Namespace,
    global_step: int,
    epoch: int,
    weight_config: dict[str, Any],
    vocab_size: int,
    key_labels_available: bool,
    checkpoint_path: str,
) -> dict[str, Any]:
    sacrebleu = get_sacrebleu()
    model.eval()
    predictions, skipped_audio_errors = generate_predictions(
        model=model,
        processor=processor,
        samples=samples,
        accelerator=accelerator,
        batch_size=args.eval_batch_size,
        max_new_tokens=args.val_max_new_tokens or args.max_new_tokens,
        sampling_rate=args.sampling_rate,
        args=args,
    )
    references = [sample["reference"] for sample in samples]
    eval_bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="zh").score
    eval_chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=0).score
    eval_len_ratio = sum(
        compute_len_ratio(prediction, reference)
        for prediction, reference in zip(predictions, references)
    ) / len(samples)

    entity_rewarder = weight_config.get("_entity_rewarder")
    if key_labels_available:
        _has_any_keys = any(s.get("gold_keys") for s in samples)
    else:
        _has_any_keys = False
    if _has_any_keys:
        eval_key_recall = sum(
            compute_key_recall(
                prediction,
                sample.get("gold_keys", []),
                match_mode=weight_config["key_match_mode"],
            )
            for prediction, sample in zip(predictions, samples)
        ) / len(samples)
    else:
        eval_key_recall = None  # N/A — val data has no gold_keys

    if weight_config["normalized"]["key"] > 0 and entity_rewarder is not None:
        eval_entity_score = sum(
            entity_rewarder.score(prediction, sample)[0]
            for prediction, sample in zip(predictions, samples)
        ) / len(samples)
    else:
        eval_entity_score = eval_key_recall if eval_key_recall is not None else 0.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    eval_loss = None
    inference_mass = (
        weight_config["normalized"]["bleu"]
        + weight_config["normalized"]["chrf"]
        + weight_config["normalized"]["key"]
    )
    if inference_mass > 0:
        selection_score = (
            (weight_config["normalized"]["bleu"] / inference_mass) * (eval_bleu / 100.0)
            + (weight_config["normalized"]["chrf"] / inference_mass) * (eval_chrf / 100.0)
            + (weight_config["normalized"]["key"] / inference_mass) * eval_entity_score
        )
    else:
        selection_score = eval_bleu / 100.0

    return {
        "step": global_step,
        "epoch": epoch,
        "objective": args.objective,
        "eval_bleu": eval_bleu,
        "eval_chrf": eval_chrf,
        "eval_key_recall": eval_key_recall,
        "eval_entity_score": eval_entity_score,
        "eval_len_ratio": eval_len_ratio,
        "eval_loss": eval_loss,
        "selection_score": selection_score,
        "selection_mode": "inference_metrics_only",
        "bleu_weight": weight_config["normalized"]["bleu"],
        "chrf_weight": weight_config["normalized"]["chrf"],
        "key_weight": weight_config["normalized"]["key"],
        "ce_weight": weight_config["normalized"]["ce"],
        "checkpoint_path": checkpoint_path,
        "num_samples": len(samples),
        "skipped_audio_errors": skipped_audio_errors,
        "key_labels_available": key_labels_available,
        "entity_reward_mode": weight_config.get("entity_reward_mode", "key_recall"),
        "entity_soft_tau": weight_config.get("entity_soft_tau"),
    }


def save_adapter(model: torch.nn.Module, output_dir: Path, accelerator: Accelerator) -> None:
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        staging_dir = output_dir.with_name(f"{output_dir.name}.tmp")
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        staging_dir.mkdir(parents=True, exist_ok=True)
        accelerator.unwrap_model(model).save_pretrained(staging_dir)
        if output_dir.exists():
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
    accelerator: Accelerator,
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
    save_adapter(model, staging_dir / "policy_adapter", accelerator)

    if accelerator.is_main_process:
        trainer_state = {
            "checkpoint_version": 1,
            "global_step": global_step,
            "next_epoch": next_epoch,
            "next_batch_index": next_batch_index,
            "best_score": best_score,
            "skipped_nonfinite_steps": skipped_nonfinite_steps,
            "skipped_audio_errors": skipped_audio_errors,
            "init_adapter_path": str(args.init_adapter_path) if args.init_adapter_path else None,
            "reason": reason,
            "max_train_steps": max_train_steps,
            "num_train_epochs": args.num_train_epochs,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "kl_coef": args.kl_coef,
        }
        save_json(staging_dir / "trainer_state.json", trainer_state)
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
        staging_dir.replace(checkpoint_dir)
        write_latest_checkpoint_marker(checkpoint_root, checkpoint_dir)
        prune_old_checkpoints(checkpoint_root, args.keep_last_checkpoints)
    accelerator.wait_for_everyone()
    return checkpoint_dir


def checkpoint_next_position(epoch: int, batch_index: int, num_epoch_batches: int) -> tuple[int, int]:
    if batch_index >= num_epoch_batches:
        return epoch + 1, 1
    return epoch, batch_index + 1


def shuffled_epoch_samples(samples: list[dict[str, Any]], seed: int, epoch: int) -> list[dict[str, Any]]:
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
    if args.num_candidates <= 0:
        raise ValueError("--num-candidates must be >= 1.")
    if args.kl_coef < 0:
        raise ValueError("--kl-coef must be >= 0.")
    if args.kl_coef > 0 and args.init_adapter_path is None:
        raise ValueError("--kl-coef > 0 requires --init-adapter-path as the frozen KL reference.")
    if args.checkpoint_every_steps < 0:
        raise ValueError("--checkpoint-every-steps must be >= 0.")
    if args.use_deepspeed_zero2 and args.use_deepspeed_zero3:
        raise ValueError("--use-deepspeed-zero2 and --use-deepspeed-zero3 cannot both be enabled.")
    if args.load_in_4bit and args.use_deepspeed_zero3:
        raise ValueError("--load-in-4bit and --use-deepspeed-zero3 should not be used together.")
    if args.load_in_8bit and args.use_deepspeed_zero3:
        raise ValueError("--load-in-8bit and --use-deepspeed-zero3 should not be used together.")
    if args.torchao_int8_weight_only and args.use_deepspeed_zero3:
        raise ValueError("--torchao-int8-weight-only and --use-deepspeed-zero3 should not be used together.")
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("--load-in-8bit and --load-in-4bit should not be used together.")
    if args.torchao_int8_weight_only and args.load_in_4bit:
        raise ValueError("--torchao-int8-weight-only and --load-in-4bit should not be used together.")
    if args.torchao_int8_weight_only and args.load_in_8bit:
        raise ValueError("--torchao-int8-weight-only and --load-in-8bit should not be used together.")
    if args.init_adapter_path is not None and not args.init_adapter_path.exists():
        raise ValueError(f"--init-adapter-path does not exist: {args.init_adapter_path}")

    weight_config = resolve_weight_config(args)
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
        saved_init_adapter = resume_trainer_state.get("init_adapter_path")
        current_init_adapter = str(args.init_adapter_path) if args.init_adapter_path else None
        if saved_init_adapter != current_init_adapter:
            raise ValueError(
                "Resume checkpoint was created with a different init adapter. "
                f"checkpoint={saved_init_adapter}, current={current_init_adapter}"
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
                "method": "EARL-GRPO",
                "runtime_impl": "fca_grpo_risk_runtime",
                "objective_impl": "kl_regularized_group_relative_risk",
                "grpo_objective": args.grpo_objective,
                "base_model_path": args.base_model_path,
                "init_adapter_path": str(args.init_adapter_path) if args.init_adapter_path else None,
                "train_data_path": str(args.train_data_path),
                "val_data_path": str(args.val_data_path),
                "train_entity_path": str(args.train_entity_path) if args.train_entity_path else None,
                "val_entity_path": str(args.val_entity_path) if args.val_entity_path else None,
                "objective": args.objective,
                "raw_weights": weight_config["raw"],
                "normalized_weights": weight_config["normalized"],
                "normalized_reward_weights": weight_config["reward_weights"],
                "risk_objective_weight": weight_config["risk_objective_weight"],
                "group_policy_scale": args.group_policy_scale,
                "reward_normalization": "zscore_per_sample_candidates",
                "key_match_mode": args.key_match_mode,
                "entity_reward_mode": args.entity_reward_mode,
                "entity_soft_tau": args.entity_soft_tau,
                "entity_embedding_model": args.entity_embedding_model,
                "entity_embedding_pooling": args.entity_embedding_pooling,
                "ner_tokenizer_model": args.ner_tokenizer_model,
                "ner_model": args.ner_model,
                "ce_loss_normalization": "clip(ce_loss / ln(vocab_size), 0, 1)",
                "output_dir": str(experiment_dir),
                "best_checkpoint_path": str(adapter_best_dir),
                "last_checkpoint_path": str(adapter_last_dir),
                "tensorboard_dir": str(tensorboard_dir) if args.enable_tensorboard else None,
                "generation_strategy": args.generation_strategy,
                "diversity_penalty": args.diversity_penalty,
                "length_penalty": args.length_penalty,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "no_repeat_ngram_size": args.no_repeat_ngram_size,
                "max_new_tokens": args.max_new_tokens,
                "num_candidates": args.num_candidates,
                "mixed_precision": args.mixed_precision,
                "learning_rate": args.learning_rate,
                "kl_coef": args.kl_coef,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
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
    vocab_size = get_vocab_size(processor)
    model, target_modules, trainable_params, total_params = apply_lora(base_model, args)
    quantization_summary = collect_quantization_summary(model)

    # entity_gemma_fuzzy reuses the training model for NER (no second copy in
    # VRAM).  We disable LoRA adapters inside extract() so the NER is done by
    # the underlying base Gemma weights, not the translation-tuned policy.
    _entity_rewarder = weight_config.get("_entity_rewarder")
    if _entity_rewarder is not None and getattr(args, "entity_reward_mode", "") == "entity_gemma_fuzzy":
        _entity_rewarder.set_shared_model(model, processor)
    optimizer = AdamW(
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
    if args.gradient_checkpointing and hasattr(model, "_set_static_graph"):
        try:
            model._set_static_graph()
        except Exception:
            pass
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]

    global_step = 0
    best_score = float("-inf")
    start_epoch = 1
    start_batch_index = 1
    skipped_nonfinite_steps = 0
    skipped_audio_errors = 0
    if resume_checkpoint_dir is not None:
        accelerator.load_state(str(resume_checkpoint_dir / "accelerator_state"))
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
                    "normalized_weights": weight_config["normalized"],
                    "effective_gradient_checkpointing": weight_config["effective_gradient_checkpointing"],
                    "quantization": quantization_summary,
                    "resume_checkpoint": str(resume_checkpoint_dir) if resume_checkpoint_dir is not None else None,
                    "start_epoch": start_epoch,
                    "start_batch_index": start_batch_index,
                    "global_step": global_step,
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
    accumulated_rewards: list[float] = []
    accumulated_best_minus_mean_rewards: list[float] = []
    accumulated_reward_zscores: list[float] = []
    accumulated_reward_zscore_stds: list[float] = []
    accumulated_bleu: list[float] = []
    accumulated_chrf: list[float] = []
    accumulated_key_recall: list[float] = []
    accumulated_entity_score: list[float] = []
    accumulated_kl_loss: list[float] = []
    accumulated_ce_loss: list[float] = []
    stop_training = False
    last_epoch = max(1, min(start_epoch, args.num_train_epochs))
    for epoch in range(start_epoch, args.num_train_epochs + 1):
        last_epoch = epoch
        shuffled_samples = shuffled_epoch_samples(train_samples, args.seed, epoch)
        epoch_batches = list(iter_batches(shuffled_samples, args.per_device_train_batch_size))

        for batch_index, batch in enumerate(epoch_batches, start=1):
            if epoch == start_epoch and batch_index < start_batch_index:
                continue
            model.train()
            sample_losses: list[torch.Tensor] = []
            batch_metrics: list[dict[str, float]] = []

            for sample_index_in_batch, sample in enumerate(batch):
                weight_config["_debug_context"] = {
                    "global_step": global_step,
                    "update_step": global_step + 1,
                    "epoch": epoch,
                    "batch_index": batch_index,
                    "sample_index_in_batch": sample_index_in_batch,
                }
                try:
                    sample_loss, sample_metrics = compute_group_risk_loss_for_sample(
                        model=model,
                        processor=processor,
                        sample=sample,
                        accelerator=accelerator,
                        args=args,
                        weight_config=weight_config,
                        vocab_size=vocab_size,
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
            weight_config.pop("_debug_context", None)

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
            accumulated_policy_losses.extend(
                metric.get("policy_loss", metric.get("risk_loss", metric["loss"]))
                for metric in batch_metrics
            )
            accumulated_rewards.extend(metric["mean_reward"] for metric in batch_metrics)
            accumulated_best_minus_mean_rewards.extend(
                metric["best_minus_mean_reward"] for metric in batch_metrics
            )
            accumulated_reward_zscores.extend(metric["mean_reward_zscore"] for metric in batch_metrics)
            accumulated_reward_zscore_stds.extend(
                metric["reward_zscore_std"] for metric in batch_metrics
            )
            accumulated_bleu.extend(metric["mean_bleu"] for metric in batch_metrics)
            accumulated_chrf.extend(metric["mean_chrf"] for metric in batch_metrics)
            accumulated_key_recall.extend(metric["mean_key_recall"] for metric in batch_metrics)
            accumulated_entity_score.extend(
                metric.get("mean_entity_score", metric["mean_key_recall"]) for metric in batch_metrics
            )
            accumulated_kl_loss.extend(metric["kl_loss"] for metric in batch_metrics)
            accumulated_ce_loss.extend(metric["ce_loss"] for metric in batch_metrics)

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
                "objective_loss": sum(accumulated_policy_losses) / len(accumulated_policy_losses),
                "grpo_objective": args.grpo_objective,
                "mean_reward": sum(accumulated_rewards) / len(accumulated_rewards),
                "best_minus_mean_reward": sum(accumulated_best_minus_mean_rewards)
                / len(accumulated_best_minus_mean_rewards),
                "mean_reward_zscore": sum(accumulated_reward_zscores)
                / len(accumulated_reward_zscores),
                "reward_zscore_std": sum(accumulated_reward_zscore_stds)
                / len(accumulated_reward_zscore_stds),
                "mean_bleu": sum(accumulated_bleu) / len(accumulated_bleu),
                "mean_chrf": sum(accumulated_chrf) / len(accumulated_chrf),
                "mean_key_recall": sum(accumulated_key_recall) / len(accumulated_key_recall),
                "mean_entity_score": sum(accumulated_entity_score) / len(accumulated_entity_score),
                "mean_kl_loss": sum(accumulated_kl_loss) / len(accumulated_kl_loss),
                "kl_coef": args.kl_coef,
                "mean_ce_loss": sum(accumulated_ce_loss) / len(accumulated_ce_loss),
                "learning_rate": float(lr_scheduler.get_last_lr()[0]),
                "bleu_weight": weight_config["normalized"]["bleu"],
                "chrf_weight": weight_config["normalized"]["chrf"],
                "key_weight": weight_config["normalized"]["key"],
                "ce_weight": weight_config["normalized"]["ce"],
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
                    tb_writer.add_scalar(
                        "train/mean_reward_raw", train_row["mean_reward"], global_step
                    )
                    tb_writer.add_scalar(
                        "train/best_minus_mean_reward",
                        train_row["best_minus_mean_reward"],
                        global_step,
                    )
                    tb_writer.add_scalar(
                        "train/mean_reward_zscore", train_row["mean_reward_zscore"], global_step
                    )
                    tb_writer.add_scalar(
                        "train/reward_zscore_std", train_row["reward_zscore_std"], global_step
                    )
                    tb_writer.add_scalar("train/mean_bleu", train_row["mean_bleu"], global_step)
                    tb_writer.add_scalar("train/mean_chrf", train_row["mean_chrf"], global_step)
                    tb_writer.add_scalar("train/mean_key_recall", train_row["mean_key_recall"], global_step)
                    tb_writer.add_scalar("train/mean_entity_score", train_row["mean_entity_score"], global_step)
                    tb_writer.add_scalar("train/mean_kl_loss", train_row["mean_kl_loss"], global_step)
                    tb_writer.add_scalar("train/kl_coef", train_row["kl_coef"], global_step)
                    tb_writer.add_scalar("train/mean_ce_loss", train_row["mean_ce_loss"], global_step)
                    tb_writer.flush()
                if args.log_every_steps > 0 and global_step % args.log_every_steps == 0:
                    print(json.dumps(train_row, ensure_ascii=False))

            clear_accumulators(
                accumulated_losses,
                accumulated_policy_losses,
                accumulated_rewards,
                accumulated_best_minus_mean_rewards,
                accumulated_reward_zscores,
                accumulated_reward_zscore_stds,
                accumulated_bleu,
                accumulated_chrf,
                accumulated_key_recall,
                accumulated_entity_score,
                accumulated_kl_loss,
                accumulated_ce_loss,
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
                val_row = evaluate_validation(
                    model=model,
                    processor=processor,
                    samples=val_samples,
                    accelerator=accelerator,
                    args=args,
                    global_step=global_step,
                    epoch=epoch,
                    weight_config=weight_config,
                    vocab_size=vocab_size,
                    key_labels_available=val_key_labels_available,
                    checkpoint_path=str(adapter_best_dir),
                )
                if accelerator.is_main_process:
                    append_jsonl(val_metrics_path, val_row)
                    if tb_writer is not None:
                        tb_writer.add_scalar("val/eval_bleu", val_row["eval_bleu"], global_step)
                        tb_writer.add_scalar("val/eval_chrf", val_row["eval_chrf"], global_step)
                        if val_row["eval_key_recall"] is not None:
                            tb_writer.add_scalar("val/eval_key_recall", val_row["eval_key_recall"], global_step)
                        tb_writer.add_scalar("val/eval_entity_score", val_row["eval_entity_score"], global_step)
                        if val_row["eval_loss"] is not None:
                            tb_writer.add_scalar("val/eval_loss", val_row["eval_loss"], global_step)
                        tb_writer.add_scalar("val/selection_score", val_row["selection_score"], global_step)
                        tb_writer.flush()
                    print(json.dumps(val_row, ensure_ascii=False))
                if val_row["selection_score"] > best_score:
                    best_score = val_row["selection_score"]
                    save_adapter(model, adapter_best_dir, accelerator)
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

    final_val_row = evaluate_validation(
        model=model,
        processor=processor,
        samples=val_samples,
        accelerator=accelerator,
        args=args,
        global_step=global_step,
        epoch=last_epoch,
        weight_config=weight_config,
        vocab_size=vocab_size,
        key_labels_available=val_key_labels_available,
        checkpoint_path=str(adapter_best_dir),
    )
    if accelerator.is_main_process:
        append_jsonl(val_metrics_path, final_val_row)
        if tb_writer is not None:
            tb_writer.add_scalar("val/eval_bleu", final_val_row["eval_bleu"], global_step)
            tb_writer.add_scalar("val/eval_chrf", final_val_row["eval_chrf"], global_step)
            if final_val_row["eval_key_recall"] is not None:
                tb_writer.add_scalar("val/eval_key_recall", final_val_row["eval_key_recall"], global_step)
            tb_writer.add_scalar("val/eval_entity_score", final_val_row["eval_entity_score"], global_step)
            if final_val_row["eval_loss"] is not None:
                tb_writer.add_scalar("val/eval_loss", final_val_row["eval_loss"], global_step)
            tb_writer.add_scalar("val/selection_score", final_val_row["selection_score"], global_step)
            tb_writer.flush()
        print(json.dumps(final_val_row, ensure_ascii=False))
    if final_val_row["selection_score"] > best_score:
        best_score = final_val_row["selection_score"]
        save_adapter(model, adapter_best_dir, accelerator)

    save_adapter(model, adapter_last_dir, accelerator)
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

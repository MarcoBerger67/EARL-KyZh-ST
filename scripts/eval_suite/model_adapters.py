from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from .audio import load_audio_array
from .data import load_prediction_records
from .types import EvaluationConfig, ModelConfig, PredictionRecord, Sample


def _resolve_torch_dtype(dtype_name: str) -> str | torch.dtype:
    if dtype_name == "auto":
        return "auto"
    return getattr(torch, dtype_name)


def _get_pad_token_id(processor: Any) -> int | None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return None
    if tokenizer.pad_token_id is not None:
        return tokenizer.pad_token_id
    return tokenizer.eos_token_id


def _format_prompt(sample: Sample, config: ModelConfig) -> str:
    if config.prompt_mode == "fixed":
        return config.prompt_text.strip()
    if config.prompt_mode == "template":
        return config.prompt_template.format(
            dataset_prompt=sample.dataset_prompt,
            source_text=sample.source_text,
            reference_text=sample.reference_text,
            sample_id=sample.sample_id,
        ).strip()
    return sample.dataset_prompt.strip()


def _move_to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def _cast_floating_tensors(batch: dict[str, Any], dtype: torch.dtype) -> dict[str, Any]:
    casted: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            casted[key] = value.to(dtype=dtype)
        else:
            casted[key] = value
    return casted


def _model_dtype(model: torch.nn.Module) -> torch.dtype:
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float32


def _load_sample_audios(samples: list[Sample], sampling_rate: int) -> list[Any]:
    audios = []
    for sample in samples:
        if not sample.audio_path:
            raise ValueError(f"Sample '{sample.sample_id}' has no audio path.")
        audios.append(load_audio_array(sample.audio_path, sampling_rate))
    return audios


def _process_audio_batch(
    processor: Any,
    audios: list[Any],
    sampling_rate: int,
    padding: bool | str = True,
) -> dict[str, Any]:
    return processor(
        audio=audios,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=padding,
    )


def _load_processor_by_declared_class(processor_path: str) -> Any:
    processor_config_path = Path(processor_path) / "processor_config.json"
    tokenizer_config_path = Path(processor_path) / "tokenizer_config.json"
    declared_class = ""
    for path in (processor_config_path, tokenizer_config_path):
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        declared_class = str(payload.get("processor_class") or declared_class)
        if declared_class:
            break
    if not declared_class:
        raise ValueError(f"No processor_class found under {processor_path!r}.")

    import importlib
    import transformers

    processor_cls = getattr(transformers, declared_class, None)
    if processor_cls is None and declared_class == "Gemma4Processor":
        module = importlib.import_module("transformers.models.gemma4.processing_gemma4")
        processor_cls = getattr(module, declared_class, None)
    if processor_cls is None:
        raise ValueError(f"Declared processor class {declared_class!r} is not importable in this environment.")
    return processor_cls.from_pretrained(processor_path)


def _patch_transformers_check_model_inputs_compat() -> None:
    try:
        from transformers.utils import generic as transformers_generic
    except Exception:
        return
    original = getattr(transformers_generic, "check_model_inputs", None)
    if original is None or getattr(original, "_qwen3_asr_compat", False):
        return

    def compatible_check_model_inputs(func: Any | None = None, *args: Any, **kwargs: Any) -> Any:
        if func is None:
            try:
                return original(*args, **kwargs)
            except TypeError:
                return original
        try:
            return original(func, *args, **kwargs)
        except TypeError:
            return original(func)

    compatible_check_model_inputs._qwen3_asr_compat = True  # type: ignore[attr-defined]
    transformers_generic.check_model_inputs = compatible_check_model_inputs


def _patch_qwen3_asr_config_compat() -> None:
    try:
        from qwen_asr.core.transformers_backend import configuration_qwen3_asr
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRConfig
    except Exception:
        return
    if getattr(Qwen3ASRConfig, "_eval_suite_compat", False):
        return

    for class_name in (
        "Qwen3ASRConfig",
        "Qwen3ASRThinkerConfig",
        "Qwen3ASRThinkerTextConfig",
    ):
        config_cls = getattr(configuration_qwen3_asr, class_name, None)
        if config_cls is None:
            continue
        if not hasattr(config_cls, "pad_token_id"):
            config_cls.pad_token_id = None

    original_get_text_config = Qwen3ASRConfig.get_text_config

    def compatible_get_text_config(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return original_get_text_config(self, *args, **kwargs)
        except AttributeError as exc:
            if "thinker_config" not in str(exc):
                raise
            config_dict = getattr(self, "_sub_configs", None)
            if isinstance(config_dict, dict):
                thinker = config_dict.get("thinker_config")
                if thinker is not None and hasattr(thinker, "get_text_config"):
                    return thinker.get_text_config()
            text_config = getattr(self, "text_config", None)
            if text_config is not None:
                return text_config
            return self

    Qwen3ASRConfig.get_text_config = compatible_get_text_config
    Qwen3ASRConfig._eval_suite_compat = True


def _patch_qwen3_asr_rope_compat() -> None:
    try:
        from qwen_asr.core.transformers_backend import modeling_qwen3_asr
    except Exception:
        return
    rope_functions = getattr(modeling_qwen3_asr, "ROPE_INIT_FUNCTIONS", None)
    if not isinstance(rope_functions, dict):
        return

    try:
        from transformers import modeling_rope_utils
    except Exception:
        modeling_rope_utils = None

    default_fn = None
    if modeling_rope_utils is not None:
        default_fn = getattr(modeling_rope_utils, "_compute_default_rope_parameters", None)
        if default_fn is None:
            upstream_functions = getattr(modeling_rope_utils, "ROPE_INIT_FUNCTIONS", None)
            if isinstance(upstream_functions, dict):
                default_fn = upstream_functions.get("default")

    if default_fn is None:
        def default_fn(config: Any, device: Any, seq_len: int | None = None, **rope_kwargs: Any) -> tuple[torch.Tensor, float]:
            base = float(getattr(config, "rope_theta", 10000.0))
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None:
                hidden_size = int(getattr(config, "hidden_size"))
                num_attention_heads = int(getattr(config, "num_attention_heads"))
                head_dim = hidden_size // num_attention_heads
            partial_rotary_factor = float(getattr(config, "partial_rotary_factor", 1.0))
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                base ** (torch.arange(0, dim, 2, dtype=torch.int64, device=device).float() / dim)
            )
            return inv_freq, 1.0

    if "default" not in rope_functions:
        rope_functions["default"] = default_fn

    rotary_cls = getattr(modeling_qwen3_asr, "Qwen3ASRThinkerTextRotaryEmbedding", None)
    if rotary_cls is not None and not hasattr(rotary_cls, "compute_default_rope_parameters"):
        def compute_default_rope_parameters(
            self: Any,
            config: Any | None = None,
            device: Any | None = None,
            seq_len: int | None = None,
            **rope_kwargs: Any,
        ) -> tuple[torch.Tensor, float]:
            config = config or getattr(self, "config", None)
            if device is None:
                inv_freq = getattr(self, "inv_freq", None)
                if isinstance(inv_freq, torch.Tensor):
                    device = inv_freq.device
                else:
                    device = torch.device("cpu")
            return rope_functions["default"](config, device, seq_len=seq_len, **rope_kwargs)

        rotary_cls.compute_default_rope_parameters = compute_default_rope_parameters


def _patch_qwen3_asr_generation_compat() -> None:
    try:
        from qwen_asr.core.transformers_backend import modeling_qwen3_asr
    except Exception:
        return
    original_create_causal_mask = getattr(modeling_qwen3_asr, "create_causal_mask", None)
    if original_create_causal_mask is not None and not getattr(
        original_create_causal_mask, "_eval_suite_mask_compat", False
    ):
        try:
            mask_signature = inspect.signature(original_create_causal_mask)
            allowed_mask_kwargs = {
                name
                for name, param in mask_signature.parameters.items()
                if param.kind in {inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
            }
            accepts_var_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in mask_signature.parameters.values()
            )
        except (TypeError, ValueError):
            allowed_mask_kwargs = set()
            accepts_var_kwargs = True

        def compatible_create_causal_mask(*args: Any, **kwargs: Any) -> Any:
            if "input_embeds" in kwargs and "input_embeds" in allowed_mask_kwargs:
                pass
            elif "input_embeds" in kwargs and "inputs_embeds" in allowed_mask_kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            elif "inputs_embeds" in kwargs and "input_embeds" in allowed_mask_kwargs:
                kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
            else:
                kwargs.pop("input_embeds", None)
                kwargs.pop("inputs_embeds", None)
            if not accepts_var_kwargs:
                kwargs = {key: value for key, value in kwargs.items() if key in allowed_mask_kwargs}
            return original_create_causal_mask(*args, **kwargs)

        compatible_create_causal_mask._eval_suite_mask_compat = True  # type: ignore[attr-defined]
        modeling_qwen3_asr.create_causal_mask = compatible_create_causal_mask

    thinker_cls = getattr(modeling_qwen3_asr, "Qwen3ASRThinkerForConditionalGeneration", None)
    if thinker_cls is None or getattr(thinker_cls, "_eval_suite_generation_compat", False):
        return
    original_prepare = thinker_cls.prepare_inputs_for_generation

    def compatible_prepare_inputs_for_generation(self: Any, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("cache_position") is None:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if isinstance(input_ids, torch.Tensor):
                seq_len = int(input_ids.shape[-1])
                kwargs["cache_position"] = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
        return original_prepare(self, *args, **kwargs)

    thinker_cls.prepare_inputs_for_generation = compatible_prepare_inputs_for_generation
    thinker_cls._eval_suite_generation_compat = True


class BaseModelAdapter(ABC):
    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    @abstractmethod
    def predict_batch(self, samples: list[Sample]) -> list[PredictionRecord]:
        raise NotImplementedError


class OfflinePredictionsAdapter(BaseModelAdapter):
    def __init__(self, config: ModelConfig, evaluation: EvaluationConfig) -> None:
        super().__init__(config)
        if evaluation.prediction_path is None:
            raise ValueError("evaluation.prediction_path is required for offline_predictions mode.")
        self.predictions = load_prediction_records(
            evaluation.prediction_path,
            evaluation.prediction_id_field,
            evaluation.prediction_text_field,
        )

    def predict_batch(self, samples: list[Sample]) -> list[PredictionRecord]:
        rows: list[PredictionRecord] = []
        for sample in samples:
            record = self.predictions.get(sample.sample_id)
            if record is None:
                raise KeyError(f"Prediction file is missing sample id '{sample.sample_id}'.")
            rows.append(
                PredictionRecord(
                    sample_id=sample.sample_id,
                    prediction_text=record.prediction_text,
                    reference_text=sample.reference_text,
                    model_name=record.model_name,
                    audio_path=sample.audio_path,
                    adapter_path=record.adapter_path,
                    dataset_prompt=sample.dataset_prompt,
                    source_text=sample.source_text,
                    intermediate_asr_text=record.intermediate_asr_text,
                    metadata=record.metadata,
                )
            )
        return rows


class GemmaAudioAdapter(BaseModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if not config.base_model_path:
            raise ValueError("model.base_model_path is required for gemma4 adapters.")
        processor_path = str(config.extra.get("processor_path") or config.base_model_path)
        try:
            self.processor = AutoProcessor.from_pretrained(processor_path)
        except ValueError as exc:
            try:
                self.processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
            except Exception as fallback_exc:
                try:
                    self.processor = _load_processor_by_declared_class(processor_path)
                except Exception as direct_exc:
                    processor_config_path = Path(processor_path) / "processor_config.json"
                    tokenizer_config_path = Path(processor_path) / "tokenizer_config.json"
                    processor_debug: dict[str, Any] = {}
                    for path in (processor_config_path, tokenizer_config_path):
                        if not path.exists():
                            processor_debug[path.name] = "missing"
                            continue
                        try:
                            payload = json.loads(path.read_text(encoding="utf-8"))
                        except Exception as read_exc:  # pragma: no cover
                            processor_debug[path.name] = f"unreadable: {read_exc}"
                            continue
                        processor_debug[path.name] = {
                            key: payload.get(key)
                            for key in (
                                "processor_class",
                                "feature_extractor_type",
                                "image_processor_type",
                                "tokenizer_class",
                                "auto_map",
                            )
                            if key in payload
                        }
                    detail = json.dumps(processor_debug, ensure_ascii=False, sort_keys=True)
                    raise ValueError(
                        "Failed to load Gemma processor. The processor/tokenizer files exist, but this Transformers "
                        "runtime cannot instantiate the declared processor class. "
                        f"Tried processor_path={processor_path!r}. processor_debug={detail}. "
                        f"Direct processor import also failed: {direct_exc}. "
                        "Use a processor directory compatible with the installed Transformers version, or update the "
                        "runtime Transformers package."
                    ) from direct_exc
        except Exception as exc:
            raise ValueError(
                f"Failed to load Gemma processor from processor_path={processor_path!r}: {exc}"
            ) from exc
        model_kwargs: dict[str, Any] = {
            "torch_dtype": _resolve_torch_dtype(config.torch_dtype),
            "low_cpu_mem_usage": True,
        }
        if config.attn_implementation:
            model_kwargs["attn_implementation"] = config.attn_implementation
        if config.device_map.lower() != "none":
            model_kwargs["device_map"] = config.device_map
        model = AutoModelForImageTextToText.from_pretrained(config.base_model_path, **model_kwargs)
        if config.adapter_path:
            try:
                from peft import PeftModel
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError(
                    "peft is required to load Gemma LoRA adapters. Install it with `pip install peft`."
                ) from exc
            model = PeftModel.from_pretrained(model, config.adapter_path, is_trainable=False)
        if config.device_map.lower() == "none":
            model.to(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        self.model = model

    def predict_batch(self, samples: list[Sample]) -> list[PredictionRecord]:
        conversations = []
        for sample in samples:
            prompt = _format_prompt(sample, self.config)
            content = [{"type": "audio", "audio": load_audio_array(sample.audio_path, self.config.sampling_rate)}]
            if prompt:
                content.append({"type": "text", "text": prompt})
            conversations.append([{"role": "user", "content": content}])

        model_inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            processor_kwargs={"sampling_rate": self.config.sampling_rate},
        )
        if self.config.device_map.lower() == "none":
            model_inputs = _move_to_device(
                model_inputs,
                self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            )

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "pad_token_id": _get_pad_token_id(self.processor),
        }
        if self.config.do_sample:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p
        generation_kwargs.update(self.config.generation_kwargs)

        with torch.inference_mode():
            generated = self.model.generate(**model_inputs, **generation_kwargs)

        prompt_length = model_inputs["input_ids"].shape[1]
        decoded = self.processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [
            PredictionRecord(
                sample_id=sample.sample_id,
                prediction_text=str(text).strip(),
                reference_text=sample.reference_text,
                model_name=self.config.name,
                audio_path=sample.audio_path,
                adapter_path=self.config.adapter_path,
                dataset_prompt=sample.dataset_prompt,
                source_text=sample.source_text,
                metadata={"model_kind": self.config.kind},
            )
            for sample, text in zip(samples, decoded)
        ]


class WhisperAdapter(BaseModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        if not config.base_model_path:
            raise ValueError("model.base_model_path is required for whisper adapters.")
        self.processor = AutoProcessor.from_pretrained(config.base_model_path)
        model_kwargs: dict[str, Any] = {"torch_dtype": _resolve_torch_dtype(config.torch_dtype)}
        if config.device_map.lower() != "none":
            model_kwargs["device_map"] = config.device_map
        model = AutoModelForSpeechSeq2Seq.from_pretrained(config.base_model_path, **model_kwargs)
        if config.device_map.lower() == "none":
            model.to(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        self.model = model

    def predict_batch(self, samples: list[Sample]) -> list[PredictionRecord]:
        audios = _load_sample_audios(samples, self.config.sampling_rate)
        model_inputs = _process_audio_batch(
            self.processor,
            audios,
            self.config.sampling_rate,
            padding="max_length",
        )
        if self.config.device_map.lower() == "none":
            model_inputs = _move_to_device(
                model_inputs,
                self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            )
        model_inputs = _cast_floating_tensors(model_inputs, _model_dtype(self.model))
        generation_kwargs: dict[str, Any] = {"max_new_tokens": self.config.max_new_tokens}
        if self.config.whisper_language:
            generation_kwargs["language"] = self.config.whisper_language
        if self.config.whisper_task:
            generation_kwargs["task"] = self.config.whisper_task
        generation_kwargs.update(self.config.generation_kwargs)
        with torch.inference_mode():
            generated = self.model.generate(**model_inputs, **generation_kwargs)
        decoded = self.processor.batch_decode(generated, skip_special_tokens=True)
        return [
            PredictionRecord(
                sample_id=sample.sample_id,
                prediction_text=str(text).strip(),
                reference_text=sample.reference_text,
                model_name=self.config.name,
                audio_path=sample.audio_path,
                dataset_prompt=sample.dataset_prompt,
                source_text=sample.source_text,
                metadata={"model_kind": self.config.kind, "whisper_task": self.config.whisper_task},
            )
            for sample, text in zip(samples, decoded)
        ]


class SeamlessM4TAdapter(BaseModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

        if not config.base_model_path:
            raise ValueError("model.base_model_path is required for seamless adapters.")
        self.processor = AutoProcessor.from_pretrained(config.base_model_path)
        model_kwargs: dict[str, Any] = {"torch_dtype": _resolve_torch_dtype(config.torch_dtype)}
        if config.device_map.lower() != "none":
            model_kwargs["device_map"] = config.device_map
        try:
            model = SeamlessM4Tv2ForSpeechToText.from_pretrained(config.base_model_path, **model_kwargs)
        except TypeError:
            dtype = model_kwargs.pop("torch_dtype", None)
            if dtype is not None:
                model_kwargs["dtype"] = dtype
            model = SeamlessM4Tv2ForSpeechToText.from_pretrained(config.base_model_path, **model_kwargs)
        if config.adapter_path:
            try:
                from peft import PeftModel
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError(
                    "peft is required to load SeamlessM4T LoRA adapters. Install it with `pip install peft`."
                ) from exc
            model = PeftModel.from_pretrained(model, config.adapter_path, is_trainable=False)
        if config.device_map.lower() == "none":
            model.to(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        self.model = model

    @staticmethod
    def _extract_text_token_ids(generated: Any) -> torch.Tensor:
        if hasattr(generated, "sequences"):
            generated = generated.sequences
        elif isinstance(generated, dict) and "sequences" in generated:
            generated = generated["sequences"]
        elif isinstance(generated, (tuple, list)):
            int_tensors = [
                item
                for item in generated
                if isinstance(item, torch.Tensor) and item.dtype in {torch.int8, torch.int16, torch.int32, torch.int64}
            ]
            if int_tensors:
                generated = int_tensors[0]
            else:
                generated = generated[0]
        if not isinstance(generated, torch.Tensor):
            raise TypeError(f"Unexpected SeamlessM4T generate output type: {type(generated)!r}")
        if generated.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
            raise TypeError(
                "SeamlessM4T generate output is not token ids. "
                f"Got dtype={generated.dtype}, shape={tuple(generated.shape)}. "
                "Use SeamlessM4Tv2ForSpeechToText for text translation."
            )
        if generated.ndim == 1:
            generated = generated.unsqueeze(0)
        return generated

    def predict_batch(self, samples: list[Sample]) -> list[PredictionRecord]:
        if not self.config.seamless_target_lang:
            raise ValueError("model.seamless_target_lang is required for seamless_s2tt.")
        audios = _load_sample_audios(samples, self.config.sampling_rate)
        model_inputs = _process_audio_batch(self.processor, audios, self.config.sampling_rate)
        if self.config.device_map.lower() == "none":
            model_inputs = _move_to_device(
                model_inputs,
                self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            )
        model_inputs = _cast_floating_tensors(model_inputs, _model_dtype(self.model))
        generation_kwargs: dict[str, Any] = {
            "tgt_lang": self.config.seamless_target_lang,
            "max_new_tokens": self.config.max_new_tokens,
        }
        generation_kwargs.update(self.config.generation_kwargs)
        with torch.inference_mode():
            generated = self.model.generate(**model_inputs, **generation_kwargs)
        token_ids = self._extract_text_token_ids(generated)
        decoded = self.processor.batch_decode(token_ids, skip_special_tokens=True)
        return [
            PredictionRecord(
                sample_id=sample.sample_id,
                prediction_text=str(text).strip(),
                reference_text=sample.reference_text,
                model_name=self.config.name,
                audio_path=sample.audio_path,
                adapter_path=self.config.adapter_path,
                dataset_prompt=sample.dataset_prompt,
                source_text=sample.source_text,
                metadata={
                    "model_kind": self.config.kind,
                    "tgt_lang": self.config.seamless_target_lang,
                    "adapter_path": self.config.adapter_path,
                },
            )
            for sample, text in zip(samples, decoded)
        ]


class Qwen2AudioAdapter(BaseModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        if not config.base_model_path:
            raise ValueError("model.base_model_path is required for Qwen2-Audio adapters.")
        self.processor = AutoProcessor.from_pretrained(config.base_model_path)
        model_kwargs: dict[str, Any] = {"torch_dtype": _resolve_torch_dtype(config.torch_dtype)}
        if config.device_map.lower() != "none":
            model_kwargs["device_map"] = config.device_map
        model = Qwen2AudioForConditionalGeneration.from_pretrained(config.base_model_path, **model_kwargs)
        if config.device_map.lower() == "none":
            model.to(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        self.model = model

    def predict_batch(self, samples: list[Sample]) -> list[PredictionRecord]:
        conversations = []
        audios = _load_sample_audios(samples, self.config.sampling_rate)
        for sample in samples:
            prompt = _format_prompt(sample, self.config)
            content = [{"type": "audio", "audio_url": sample.audio_path}]
            if prompt:
                content.append({"type": "text", "text": prompt})
            conversations.append([{"role": "user", "content": content}])

        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        model_inputs = self.processor(
            text=text,
            audios=audios,
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        if self.config.device_map.lower() == "none":
            model_inputs = _move_to_device(
                model_inputs,
                self.config.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            )
        model_inputs = _cast_floating_tensors(model_inputs, _model_dtype(self.model))

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
        }
        if self.config.do_sample:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p
        generation_kwargs.update(self.config.generation_kwargs)

        with torch.inference_mode():
            generated = self.model.generate(**model_inputs, **generation_kwargs)

        prompt_length = model_inputs["input_ids"].shape[1]
        decoded = self.processor.batch_decode(
            generated[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [
            PredictionRecord(
                sample_id=sample.sample_id,
                prediction_text=str(text).strip(),
                reference_text=sample.reference_text,
                model_name=self.config.name,
                audio_path=sample.audio_path,
                dataset_prompt=sample.dataset_prompt,
                source_text=sample.source_text,
                metadata={"model_kind": self.config.kind},
            )
            for sample, text in zip(samples, decoded)
        ]


class Qwen25OmniAdapter(BaseModelAdapter):
    DEFAULT_SYSTEM_PROMPT = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech."
    )

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        try:
            from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

            model_class = Qwen2_5OmniThinkerForConditionalGeneration
            self.full_omni = False
        except ImportError:
            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

            model_class = Qwen2_5OmniForConditionalGeneration
            self.full_omni = True

        if not config.base_model_path:
            raise ValueError("model.base_model_path is required for Qwen2.5-Omni adapters.")
        self.processor = Qwen2_5OmniProcessor.from_pretrained(config.base_model_path)
        model_kwargs: dict[str, Any] = {"torch_dtype": _resolve_torch_dtype(config.torch_dtype)}
        if self.full_omni:
            model_kwargs["enable_audio_output"] = False
        if config.device_map.lower() != "none":
            model_kwargs["device_map"] = config.device_map
        try:
            model = model_class.from_pretrained(config.base_model_path, **model_kwargs)
        except TypeError:
            dtype = model_kwargs.pop("torch_dtype", None)
            if dtype is not None:
                model_kwargs["dtype"] = dtype
            model = model_class.from_pretrained(config.base_model_path, **model_kwargs)
        if config.adapter_path:
            try:
                from peft import PeftModel
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError(
                    "peft is required to load Qwen2.5-Omni LoRA adapters. Install it with `pip install peft`."
                ) from exc
            model = PeftModel.from_pretrained(model, config.adapter_path, is_trainable=False)
        if config.device_map.lower() == "none":
            model.to(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model.eval()
        self.model = model

    def _target_device(self) -> str:
        if self.config.device_map.lower() == "none":
            return self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        thinker = getattr(self.model, "thinker", None)
        if thinker is not None and hasattr(thinker, "device"):
            return str(thinker.device)
        if hasattr(self.model, "device"):
            return str(self.model.device)
        return str(next(self.model.parameters()).device)

    def predict_batch(self, samples: list[Sample]) -> list[PredictionRecord]:
        conversations = []
        audios = _load_sample_audios(samples, self.config.sampling_rate)
        for sample in samples:
            if not sample.audio_path:
                raise ValueError(f"Sample '{sample.sample_id}' has no audio path.")
            prompt = _format_prompt(sample, self.config)
            content = [{"type": "audio", "audio": sample.audio_path}]
            if prompt:
                content.append({"type": "text", "text": prompt})
            conversations.append(
                [
                    {"role": "system", "content": [{"type": "text", "text": self.DEFAULT_SYSTEM_PROMPT}]},
                    {"role": "user", "content": content},
                ]
            )

        text = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=False,
        )
        model_inputs = self.processor(
            text=text,
            audio=audios,
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(self._target_device())
        else:
            model_inputs = _move_to_device(model_inputs, self._target_device())
        model_inputs = _cast_floating_tensors(model_inputs, _model_dtype(self.model))

        generation_kwargs = {
            "max_new_tokens": self.config.max_new_tokens,
            "do_sample": self.config.do_sample,
            "use_audio_in_video": False,
        }
        if self.full_omni:
            generation_kwargs["return_audio"] = False
        if self.config.do_sample:
            generation_kwargs["temperature"] = self.config.temperature
            generation_kwargs["top_p"] = self.config.top_p
        generation_kwargs.update(self.config.generation_kwargs)

        with torch.inference_mode():
            generated = self.model.generate(**model_inputs, **generation_kwargs)
        if isinstance(generated, tuple):
            generated = generated[0]

        prompt_length = model_inputs["input_ids"].shape[1] if "input_ids" in model_inputs else 0
        if prompt_length and getattr(generated, "ndim", 0) == 2 and generated.shape[1] > prompt_length:
            generated = generated[:, prompt_length:]
        decoded = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [
            PredictionRecord(
                sample_id=sample.sample_id,
                prediction_text=str(text).strip(),
                reference_text=sample.reference_text,
                model_name=self.config.name,
                audio_path=sample.audio_path,
                adapter_path=self.config.adapter_path,
                dataset_prompt=sample.dataset_prompt,
                source_text=sample.source_text,
                metadata={"model_kind": self.config.kind, "adapter_path": self.config.adapter_path},
            )
            for sample, text in zip(samples, decoded)
        ]


class WhisperNLLBCascadeAdapter(BaseModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        whisper_config = ModelConfig(**{**asdict(config), "kind": "whisper_asr"})
        self.whisper = WhisperAdapter(whisper_config)

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        nllb_model_path = str(config.extra.get("nllb_model_path", "facebook/nllb-200-distilled-600M"))
        self.nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_path)
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
            nllb_model_path,
            torch_dtype=_resolve_torch_dtype(config.torch_dtype),
        )
        nllb_adapter_path = config.extra.get("nllb_adapter_path")
        if nllb_adapter_path:
            try:
                from peft import PeftModel
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError(
                    "peft is required to load the NLLB LoRA adapter. Install it with `pip install peft`."
                ) from exc
            self.nllb_model = PeftModel.from_pretrained(
                self.nllb_model,
                str(nllb_adapter_path),
                is_trainable=False,
            )
        self.translation_device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.nllb_model.to(self.translation_device)
        self.nllb_model.eval()

    def _translate_batch(self, texts: list[str]) -> list[str]:
        if not self.config.nllb_target_lang:
            raise ValueError("model.nllb_target_lang is required for whisper_nllb_cascade.")
        if self.config.nllb_source_lang:
            self.nllb_tokenizer.src_lang = self.config.nllb_source_lang
        model_inputs = self.nllb_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        model_inputs = _move_to_device(model_inputs, self.translation_device)
        forced_bos = self.nllb_tokenizer.convert_tokens_to_ids(self.config.nllb_target_lang)
        with torch.inference_mode():
            generated = self.nllb_model.generate(
                **model_inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=self.config.max_new_tokens,
            )
        return self.nllb_tokenizer.batch_decode(generated, skip_special_tokens=True)

    def predict_batch(self, samples: list[Sample]) -> list[PredictionRecord]:
        asr_rows = self.whisper.predict_batch(samples)
        translated = self._translate_batch([row.prediction_text for row in asr_rows])
        rows: list[PredictionRecord] = []
        for sample, asr_row, translation in zip(samples, asr_rows, translated):
            rows.append(
                PredictionRecord(
                    sample_id=sample.sample_id,
                    prediction_text=str(translation).strip(),
                    reference_text=sample.reference_text,
                    model_name=self.config.name,
                    audio_path=sample.audio_path,
                    dataset_prompt=sample.dataset_prompt,
                    source_text=sample.source_text,
                    intermediate_asr_text=asr_row.prediction_text,
                    metadata={
                        "model_kind": self.config.kind,
                        "whisper_task": self.config.whisper_task,
                        "nllb_source_lang": self.config.nllb_source_lang,
                        "nllb_target_lang": self.config.nllb_target_lang,
                        "nllb_adapter_path": self.config.extra.get("nllb_adapter_path"),
                    },
                )
            )
        return rows


class Wav2Vec2CTCNLLBCascadeAdapter(BaseModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        from transformers import AutoModelForCTC, AutoModelForSeq2SeqLM, AutoProcessor, AutoTokenizer

        if not config.base_model_path:
            raise ValueError("model.base_model_path is required for wav2vec2_ctc_nllb_cascade.")
        self.asr_processor = AutoProcessor.from_pretrained(config.base_model_path)
        asr_kwargs: dict[str, Any] = {"torch_dtype": _resolve_torch_dtype(config.torch_dtype)}
        if config.device_map.lower() != "none":
            asr_kwargs["device_map"] = config.device_map
        try:
            asr_model = AutoModelForCTC.from_pretrained(config.base_model_path, **asr_kwargs)
        except TypeError:
            dtype = asr_kwargs.pop("torch_dtype", None)
            if dtype is not None:
                asr_kwargs["dtype"] = dtype
            asr_model = AutoModelForCTC.from_pretrained(config.base_model_path, **asr_kwargs)

        asr_adapter_path = config.adapter_path or config.extra.get("asr_adapter_path")
        if asr_adapter_path:
            try:
                from peft import PeftModel
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError(
                    "peft is required to load the Wav2Vec2 ASR LoRA adapter. Install it with `pip install peft`."
                ) from exc
            asr_model = PeftModel.from_pretrained(asr_model, str(asr_adapter_path), is_trainable=False)
        if config.device_map.lower() == "none":
            asr_model.to(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
        asr_model.eval()
        self.asr_model = asr_model

        nllb_model_path = str(config.extra.get("nllb_model_path", "facebook/nllb-200-3.3B"))
        self.nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_path)
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
            nllb_model_path,
            torch_dtype=_resolve_torch_dtype(config.torch_dtype),
        )
        nllb_adapter_path = config.extra.get("nllb_adapter_path")
        if nllb_adapter_path:
            try:
                from peft import PeftModel
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError(
                    "peft is required to load the NLLB LoRA adapter. Install it with `pip install peft`."
                ) from exc
            self.nllb_model = PeftModel.from_pretrained(
                self.nllb_model,
                str(nllb_adapter_path),
                is_trainable=False,
            )
        self.asr_adapter_path = str(asr_adapter_path) if asr_adapter_path else None
        self.nllb_adapter_path = str(nllb_adapter_path) if nllb_adapter_path else None
        self.asr_device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.translation_device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.nllb_model.to(self.translation_device)
        self.nllb_model.eval()

    def _transcribe_batch(self, samples: list[Sample]) -> list[str]:
        audios = _load_sample_audios(samples, self.config.sampling_rate)
        model_inputs = self.asr_processor(
            audios,
            sampling_rate=self.config.sampling_rate,
            return_tensors="pt",
            padding=True,
        )
        if self.config.device_map.lower() == "none":
            model_inputs = _move_to_device(model_inputs, self.asr_device)
        model_inputs = _cast_floating_tensors(model_inputs, _model_dtype(self.asr_model))
        with torch.inference_mode():
            logits = self.asr_model(**model_inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return [
            str(text).strip()
            for text in self.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)
        ]

    def _translate_batch(self, texts: list[str]) -> list[str]:
        if not self.config.nllb_target_lang:
            raise ValueError("model.nllb_target_lang is required for wav2vec2_ctc_nllb_cascade.")
        if self.config.nllb_source_lang:
            self.nllb_tokenizer.src_lang = self.config.nllb_source_lang
        model_inputs = self.nllb_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        model_inputs = _move_to_device(model_inputs, self.translation_device)
        forced_bos = self.nllb_tokenizer.convert_tokens_to_ids(self.config.nllb_target_lang)
        generation_kwargs = {
            "forced_bos_token_id": forced_bos,
            "max_new_tokens": self.config.max_new_tokens,
        }
        generation_kwargs.update(self.config.generation_kwargs)
        with torch.inference_mode():
            generated = self.nllb_model.generate(**model_inputs, **generation_kwargs)
        return self.nllb_tokenizer.batch_decode(generated, skip_special_tokens=True)

    def predict_batch(self, samples: list[Sample]) -> list[PredictionRecord]:
        asr_texts = self._transcribe_batch(samples)
        translated = self._translate_batch(asr_texts)
        rows: list[PredictionRecord] = []
        for sample, asr_text, translation in zip(samples, asr_texts, translated):
            rows.append(
                PredictionRecord(
                    sample_id=sample.sample_id,
                    prediction_text=str(translation).strip(),
                    reference_text=sample.reference_text,
                    model_name=self.config.name,
                    audio_path=sample.audio_path,
                    adapter_path=self.asr_adapter_path,
                    dataset_prompt=sample.dataset_prompt,
                    source_text=sample.source_text,
                    intermediate_asr_text=asr_text,
                    metadata={
                        "model_kind": self.config.kind,
                        "asr_adapter_path": self.asr_adapter_path,
                        "nllb_model_path": self.config.extra.get("nllb_model_path"),
                        "nllb_adapter_path": self.nllb_adapter_path,
                        "nllb_source_lang": self.config.nllb_source_lang,
                        "nllb_target_lang": self.config.nllb_target_lang,
                    },
                )
            )
        return rows


class Qwen3ASRNLLBCascadeAdapter(BaseModelAdapter):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        _patch_transformers_check_model_inputs_compat()
        try:
            from qwen_asr import Qwen3ASRModel
            _patch_qwen3_asr_config_compat()
            _patch_qwen3_asr_rope_compat()
            _patch_qwen3_asr_generation_compat()
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "qwen-asr is required for Qwen3-ASR evaluation. Install it in the runtime env, "
                "or use the official Qwen3-ASR environment."
            ) from exc

        model_kwargs: dict[str, Any] = {
            "dtype": _resolve_torch_dtype(config.torch_dtype),
            "device_map": config.device_map,
            "max_inference_batch_size": int(config.extra.get("asr_batch_size", config.batch_size)),
            "max_new_tokens": int(config.extra.get("asr_max_new_tokens", 256)),
        }
        asr_attn_implementation = config.extra.get("asr_attn_implementation", "eager")
        if asr_attn_implementation:
            model_kwargs["attn_implementation"] = str(asr_attn_implementation)
        try:
            self.asr_model = Qwen3ASRModel.from_pretrained(config.base_model_path, **model_kwargs)
        except TypeError:
            model_kwargs.pop("max_inference_batch_size", None)
            model_kwargs.pop("max_new_tokens", None)
            self.asr_model = Qwen3ASRModel.from_pretrained(config.base_model_path, **model_kwargs)
        self.asr_language = config.extra.get("asr_language")

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        nllb_model_path = str(config.extra.get("nllb_model_path", "facebook/nllb-200-distilled-600M"))
        self.nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_path)
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(
            nllb_model_path,
            torch_dtype=_resolve_torch_dtype(config.torch_dtype),
        )
        nllb_adapter_path = config.extra.get("nllb_adapter_path")
        if nllb_adapter_path:
            try:
                from peft import PeftModel
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise RuntimeError(
                    "peft is required to load the NLLB LoRA adapter. Install it with `pip install peft`."
                ) from exc
            self.nllb_model = PeftModel.from_pretrained(
                self.nllb_model,
                str(nllb_adapter_path),
                is_trainable=False,
            )
        self.nllb_adapter_path = str(nllb_adapter_path) if nllb_adapter_path else None
        self.translation_device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.nllb_model.to(self.translation_device)
        self.nllb_model.eval()

    def _transcribe_batch(self, samples: list[Sample]) -> list[str]:
        paths = []
        for sample in samples:
            if not sample.audio_path:
                raise ValueError(f"Sample '{sample.sample_id}' has no audio path.")
            paths.append(sample.audio_path)
        result = self.asr_model.transcribe(
            audio=paths,
            language=self.asr_language,
        )
        if isinstance(result, str):
            return [result]
        texts = []
        for item in result:
            if isinstance(item, str):
                texts.append(item)
            elif isinstance(item, dict):
                texts.append(str(item.get("text", item.get("transcript", ""))).strip())
            else:
                texts.append(str(getattr(item, "text", item)).strip())
        return texts

    def _translate_batch(self, texts: list[str]) -> list[str]:
        if not self.config.nllb_target_lang:
            raise ValueError("model.nllb_target_lang is required for qwen3_asr_nllb_cascade.")
        if self.config.nllb_source_lang:
            self.nllb_tokenizer.src_lang = self.config.nllb_source_lang
        model_inputs = self.nllb_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        model_inputs = _move_to_device(model_inputs, self.translation_device)
        forced_bos = self.nllb_tokenizer.convert_tokens_to_ids(self.config.nllb_target_lang)
        with torch.inference_mode():
            generated = self.nllb_model.generate(
                **model_inputs,
                forced_bos_token_id=forced_bos,
                max_new_tokens=self.config.max_new_tokens,
            )
        return self.nllb_tokenizer.batch_decode(generated, skip_special_tokens=True)

    def predict_batch(self, samples: list[Sample]) -> list[PredictionRecord]:
        asr_texts = self._transcribe_batch(samples)
        translated = self._translate_batch(asr_texts)
        rows: list[PredictionRecord] = []
        for sample, asr_text, translation in zip(samples, asr_texts, translated):
            rows.append(
                PredictionRecord(
                    sample_id=sample.sample_id,
                    prediction_text=str(translation).strip(),
                    reference_text=sample.reference_text,
                    model_name=self.config.name,
                    audio_path=sample.audio_path,
                    dataset_prompt=sample.dataset_prompt,
                    source_text=sample.source_text,
                    intermediate_asr_text=asr_text,
                    metadata={
                        "model_kind": self.config.kind,
                        "asr_language": self.asr_language,
                        "nllb_source_lang": self.config.nllb_source_lang,
                        "nllb_target_lang": self.config.nllb_target_lang,
                        "nllb_model_path": self.config.extra.get("nllb_model_path"),
                        "nllb_adapter_path": self.nllb_adapter_path,
                    },
                )
            )
        return rows


def build_model_adapter(config: ModelConfig, evaluation: EvaluationConfig) -> BaseModelAdapter:
    if config.kind == "offline_predictions":
        return OfflinePredictionsAdapter(config, evaluation)
    if config.kind in {"gemma4_audio", "gemma4_audio_lora"}:
        return GemmaAudioAdapter(config)
    if config.kind == "whisper_asr":
        return WhisperAdapter(config)
    if config.kind == "seamless_s2tt":
        return SeamlessM4TAdapter(config)
    if config.kind == "qwen2_audio_s2tt":
        return Qwen2AudioAdapter(config)
    if config.kind == "qwen25_omni_s2tt":
        return Qwen25OmniAdapter(config)
    if config.kind == "whisper_nllb_cascade":
        return WhisperNLLBCascadeAdapter(config)
    if config.kind == "wav2vec2_ctc_nllb_cascade":
        return Wav2Vec2CTCNLLBCascadeAdapter(config)
    if config.kind == "qwen3_asr_nllb_cascade":
        return Qwen3ASRNLLBCascadeAdapter(config)
    raise ValueError(f"Unsupported model.kind: {config.kind}")

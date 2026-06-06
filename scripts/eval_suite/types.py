from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Entity:
    text: str
    label: str


@dataclass(slots=True)
class Sample:
    sample_id: str
    reference_text: str
    audio_path: str | None = None
    dataset_prompt: str = ""
    source_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PredictionRecord:
    sample_id: str
    prediction_text: str
    reference_text: str
    model_name: str
    audio_path: str | None = None
    adapter_path: str | None = None
    dataset_prompt: str = ""
    source_text: str = ""
    intermediate_asr_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DatasetConfig:
    path: Path
    format: str
    id_field: str = "id"
    audio_path_field: str = "audio_path"
    prompt_field: str = "prompt"
    source_text_field: str = "source_text"
    reference_text_field: str = "reference_text"
    prediction_text_field: str = "prediction_text"
    reference_entity_path: Path | None = None
    audio_prefix_from: str = ""
    audio_prefix_to: str = ""
    limit: int | None = None


@dataclass(slots=True)
class ModelConfig:
    kind: str
    name: str
    base_model_path: str | None = None
    adapter_path: str | None = None
    device: str | None = None
    device_map: str = "auto"
    torch_dtype: str = "auto"
    attn_implementation: str | None = None
    batch_size: int = 1
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 0.95
    sampling_rate: int = 16000
    prompt_mode: str = "dataset"
    prompt_text: str = ""
    prompt_template: str = "{dataset_prompt}"
    whisper_task: str = "transcribe"
    whisper_language: str | None = None
    seamless_target_lang: str | None = None
    nllb_source_lang: str | None = None
    nllb_target_lang: str | None = None
    generation_kwargs: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationConfig:
    mode: str
    output_dir: Path
    prediction_path: Path | None = None
    metrics: list[str] = field(default_factory=lambda: ["bleu", "chrf", "entity_f1", "entity_soft"])
    entity_soft_tau: float = 0.6
    entity_lcs_skip_labels: list[str] = field(default_factory=lambda: ["PER"])
    entity_lcs_labels: list[str] = field(default_factory=lambda: ["LOC", "PER", "TERM", "NUM", "ORG", "TIME"])
    prediction_id_field: str = "id"
    prediction_text_field: str = "prediction_text"
    progress_every: int = 20


@dataclass(slots=True)
class NERConfig:
    tokenizer_model: str = "FINE_ELECTRA_SMALL_ZH"
    ner_model: str = "MSRA_NER_ELECTRA_SMALL_ZH"
    tokenizer_path: str | None = None
    ner_path: str | None = None


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = "bert-base-multilingual-cased"
    max_length: int = 128
    pooling: str = "mean"


@dataclass(slots=True)
class EvalSpec:
    dataset: DatasetConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    ner: NERConfig = field(default_factory=NERConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    raw: dict[str, Any] = field(default_factory=dict)

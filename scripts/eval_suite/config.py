from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .types import DatasetConfig, EmbeddingConfig, EvalSpec, EvaluationConfig, ModelConfig, NERConfig


def _ensure_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected '{name}' to be a mapping.")
    return value


def _resolve_path(root: Path, value: str | None) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def _as_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"Expected '{name}' to be a list of strings.")
    return value


def load_eval_spec(path: Path) -> EvalSpec:
    root = path.resolve().parents[2]
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evaluation spec must be a YAML mapping.")

    dataset_raw = _ensure_dict(payload.get("dataset"), "dataset")
    model_raw = _ensure_dict(payload.get("model"), "model")
    evaluation_raw = _ensure_dict(payload.get("evaluation"), "evaluation")
    ner_raw = _ensure_dict(payload.get("ner", {}), "ner")
    embedding_raw = _ensure_dict(payload.get("embedding", {}), "embedding")

    dataset = DatasetConfig(
        path=_resolve_path(root, dataset_raw.get("path")),
        format=str(dataset_raw.get("format", "converted_translation_jsonl")),
        id_field=str(dataset_raw.get("id_field", "id")),
        audio_path_field=str(dataset_raw.get("audio_path_field", "audio_path")),
        prompt_field=str(dataset_raw.get("prompt_field", "prompt")),
        source_text_field=str(dataset_raw.get("source_text_field", "source_text")),
        reference_text_field=str(dataset_raw.get("reference_text_field", "reference_text")),
        prediction_text_field=str(dataset_raw.get("prediction_text_field", "prediction_text")),
        reference_entity_path=_resolve_path(root, dataset_raw.get("reference_entity_path")),
        audio_prefix_from=str(dataset_raw.get("audio_prefix_from", "")),
        audio_prefix_to=str(dataset_raw.get("audio_prefix_to", "")),
        limit=dataset_raw.get("limit"),
    )
    if dataset.path is None:
        raise ValueError("dataset.path is required.")

    model_extra = {
        key: value
        for key, value in model_raw.items()
        if key
        not in {
            "kind",
            "name",
            "base_model_path",
            "adapter_path",
            "device",
            "device_map",
            "torch_dtype",
            "attn_implementation",
            "batch_size",
            "max_new_tokens",
            "do_sample",
            "temperature",
            "top_p",
            "sampling_rate",
            "prompt_mode",
            "prompt_text",
            "prompt_template",
            "whisper_task",
            "whisper_language",
            "seamless_target_lang",
            "nllb_source_lang",
            "nllb_target_lang",
            "generation_kwargs",
        }
    }
    model = ModelConfig(
        kind=str(model_raw.get("kind", "")).strip(),
        name=str(model_raw.get("name", model_raw.get("kind", ""))).strip(),
        base_model_path=model_raw.get("base_model_path"),
        adapter_path=model_raw.get("adapter_path"),
        device=model_raw.get("device"),
        device_map=str(model_raw.get("device_map", "auto")),
        torch_dtype=str(model_raw.get("torch_dtype", "auto")),
        attn_implementation=model_raw.get("attn_implementation"),
        batch_size=int(model_raw.get("batch_size", 1)),
        max_new_tokens=int(model_raw.get("max_new_tokens", 256)),
        do_sample=bool(model_raw.get("do_sample", False)),
        temperature=float(model_raw.get("temperature", 1.0)),
        top_p=float(model_raw.get("top_p", 0.95)),
        sampling_rate=int(model_raw.get("sampling_rate", 16000)),
        prompt_mode=str(model_raw.get("prompt_mode", "dataset")),
        prompt_text=str(model_raw.get("prompt_text", "")),
        prompt_template=str(model_raw.get("prompt_template", "{dataset_prompt}")),
        whisper_task=str(model_raw.get("whisper_task", "transcribe")),
        whisper_language=model_raw.get("whisper_language"),
        seamless_target_lang=model_raw.get("seamless_target_lang"),
        nllb_source_lang=model_raw.get("nllb_source_lang"),
        nllb_target_lang=model_raw.get("nllb_target_lang"),
        generation_kwargs=dict(model_raw.get("generation_kwargs", {})),
        extra=model_extra,
    )
    if not model.kind:
        raise ValueError("model.kind is required.")

    evaluation = EvaluationConfig(
        mode=str(evaluation_raw.get("mode", "generate_and_score")),
        output_dir=_resolve_path(root, evaluation_raw.get("output_dir")),
        prediction_path=_resolve_path(root, evaluation_raw.get("prediction_path")),
        metrics=_as_list(
            evaluation_raw.get("metrics", ["bleu", "chrf", "entity_f1", "entity_soft"]),
            "evaluation.metrics",
        )
        or ["bleu", "chrf", "entity_f1", "entity_soft"],
        entity_soft_tau=float(evaluation_raw.get("entity_soft_tau", 0.6)),
        entity_lcs_skip_labels=[
            label.strip().upper()
            for label in _as_list(evaluation_raw.get("entity_lcs_skip_labels", ["PER"]), "evaluation.entity_lcs_skip_labels")
            if label.strip()
        ],
        entity_lcs_labels=[
            label.strip().upper()
            for label in _as_list(
                evaluation_raw.get("entity_lcs_labels", ["LOC", "PER", "TERM", "NUM", "ORG", "TIME"]),
                "evaluation.entity_lcs_labels",
            )
            if label.strip()
        ],
        prediction_id_field=str(evaluation_raw.get("prediction_id_field", "id")),
        prediction_text_field=str(evaluation_raw.get("prediction_text_field", "prediction_text")),
        progress_every=int(evaluation_raw.get("progress_every", 20)),
    )
    if evaluation.output_dir is None:
        raise ValueError("evaluation.output_dir is required.")

    ner = NERConfig(
        tokenizer_model=str(ner_raw.get("tokenizer_model", "FINE_ELECTRA_SMALL_ZH")),
        ner_model=str(ner_raw.get("ner_model", "MSRA_NER_ELECTRA_SMALL_ZH")),
        tokenizer_path=ner_raw.get("tokenizer_path"),
        ner_path=ner_raw.get("ner_path"),
    )

    embedding = EmbeddingConfig(
        model_name=str(embedding_raw.get("model_name", "bert-base-multilingual-cased")),
        max_length=int(embedding_raw.get("max_length", 128)),
        pooling=str(embedding_raw.get("pooling", "mean")),
    )

    return EvalSpec(
        dataset=dataset,
        model=model,
        evaluation=evaluation,
        ner=ner,
        embedding=embedding,
        raw=payload,
    )

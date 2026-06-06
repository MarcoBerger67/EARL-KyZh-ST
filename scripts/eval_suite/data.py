from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .audio import rewrite_audio_path
from .types import DatasetConfig, Entity, PredictionRecord, Sample


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = [
            str(item.get("text", "")).strip()
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(text for text in texts if text).strip()
    return ""


def _extract_audio_path_from_messages(messages: list[dict[str, Any]]) -> str | None:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "audio":
                continue
            for key in ("path", "audio", "url"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _extract_user_prompt(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _extract_text_from_content(message.get("content"))
        if text:
            return text
    return ""


def _extract_assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        text = _extract_text_from_content(message.get("content"))
        if text:
            return text
    return ""


def load_samples(config: DatasetConfig) -> list[Sample]:
    samples: list[Sample] = []
    with config.path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            sample = parse_sample_record(record, config, line_number)
            samples.append(sample)
            if config.limit is not None and len(samples) >= config.limit:
                break
    if not samples:
        raise ValueError(f"No samples were loaded from {config.path}")
    return samples


def parse_sample_record(record: dict[str, Any], config: DatasetConfig, line_number: int) -> Sample:
    if config.format == "converted_translation_jsonl":
        messages = record.get("messages")
        if not isinstance(messages, list):
            raise ValueError(f"Line {line_number}: converted_translation_jsonl requires messages.")
        audio_path = _extract_audio_path_from_messages(messages)
        if audio_path:
            audio_path = rewrite_audio_path(audio_path, config.audio_prefix_from, config.audio_prefix_to)
        return Sample(
            sample_id=str(record.get("id", f"line_{line_number}")),
            reference_text=_extract_assistant_text(messages),
            audio_path=audio_path,
            dataset_prompt=_extract_user_prompt(messages),
            source_text="",
            metadata={"raw_record": record},
        )

    if config.format == "generic_jsonl":
        audio_path = record.get(config.audio_path_field)
        if isinstance(audio_path, str) and audio_path:
            audio_path = rewrite_audio_path(audio_path, config.audio_prefix_from, config.audio_prefix_to)
        else:
            audio_path = None
        return Sample(
            sample_id=str(record.get(config.id_field, f"line_{line_number}")),
            reference_text=str(record.get(config.reference_text_field, "") or "").strip(),
            audio_path=audio_path,
            dataset_prompt=str(record.get(config.prompt_field, "") or "").strip(),
            source_text=str(record.get(config.source_text_field, "") or "").strip(),
            metadata={"raw_record": record},
        )

    raise ValueError(f"Unsupported dataset.format: {config.format}")


def load_prediction_records(
    path: Path,
    id_field: str,
    prediction_text_field: str,
    limit: int | None = None,
) -> dict[str, PredictionRecord]:
    predictions: dict[str, PredictionRecord] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            sample_id = str(record.get(id_field, f"line_{line_number}"))
            prediction_text = str(
                record.get(prediction_text_field, record.get("prediction", ""))
                or ""
            ).strip()
            predictions[sample_id] = PredictionRecord(
                sample_id=sample_id,
                prediction_text=prediction_text,
                reference_text=str(record.get("reference_text", "") or "").strip(),
                model_name=str(record.get("model_name", "offline_predictions")),
                audio_path=record.get("audio_path"),
                adapter_path=record.get("adapter_path"),
                dataset_prompt=str(record.get("dataset_prompt", "") or "").strip(),
                source_text=str(record.get("source_text", "") or "").strip(),
                intermediate_asr_text=record.get("intermediate_asr_text"),
                metadata={k: v for k, v in record.items() if k not in {id_field, prediction_text_field}},
            )
            if limit is not None and len(predictions) >= limit:
                break
    return predictions


def load_entity_sidecar(path: Path) -> dict[str, list[Entity]]:
    sidecar: dict[str, list[Entity]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            sample_id = str(record.get("id", f"line_{line_number}"))
            entities = []
            for item in record.get("entities", []):
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "") or "").strip()
                label = str(item.get("label", "") or "").strip().upper()
                if text and label:
                    entities.append(Entity(text=text, label=label))
            sidecar[sample_id] = entities
    return sidecar

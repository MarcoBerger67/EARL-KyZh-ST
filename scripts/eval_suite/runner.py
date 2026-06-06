from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import load_eval_spec
from .data import load_samples, load_entity_sidecar
from .entity_backend import (
    HanLPEntityExtractor,
    MBertEmbedder,
    cosine_similarity_matrix,
    entity_records_to_json,
)
from .io_utils import ensure_dir, write_json, write_jsonl
from .model_adapters import build_model_adapter
from .text_metrics import (
    build_key_metric_row,
    build_lcs_metric_row,
    build_per_sample_metric_row,
    compute_bleu_chrf,
    compute_exact_entity_counts,
    compute_hard_key_entity_score,
    compute_lcs_entity_score,
    compute_soft_entity_counts,
)
from .types import Entity, EvalSpec, PredictionRecord, Sample


def _progress(iterable: Any, **kwargs: Any) -> Any:
    try:
        from tqdm.auto import tqdm
    except Exception:
        return iterable
    return tqdm(iterable, **kwargs)


def _batched(items: list[Any], batch_size: int) -> list[list[Any]]:
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _predict_samples(spec: EvalSpec, samples: list[Sample]) -> list[PredictionRecord]:
    adapter = build_model_adapter(spec.model, spec.evaluation)
    rows: list[PredictionRecord] = []
    batches = _batched(samples, spec.model.batch_size)
    iterator = _progress(batches, total=len(batches), desc="Predict", unit="batch")
    for batch_index, batch in enumerate(iterator, start=1):
        rows.extend(adapter.predict_batch(batch))
        if hasattr(iterator, "set_postfix"):
            iterator.set_postfix(samples=min(batch_index * spec.model.batch_size, len(samples)))
        if spec.evaluation.progress_every > 0 and batch_index % spec.evaluation.progress_every == 0:
            print(f"Predicted {min(batch_index * spec.model.batch_size, len(samples))}/{len(samples)} samples")
    return rows


def _reference_entities(
    spec: EvalSpec,
    samples: list[Sample],
    extractor: HanLPEntityExtractor,
) -> dict[str, list[Entity]]:
    if spec.dataset.reference_entity_path and spec.dataset.reference_entity_path.exists():
        return load_entity_sidecar(spec.dataset.reference_entity_path)

    rows: dict[str, list[Entity]] = {}
    iterator = _progress(samples, total=len(samples), desc="Reference NER", unit="sample")
    for index, sample in enumerate(iterator, start=1):
        rows[sample.sample_id] = extractor.extract(sample.reference_text)
        if spec.evaluation.progress_every > 0 and index % spec.evaluation.progress_every == 0:
            print(f"Reference NER {index}/{len(samples)}")
    return rows


def _prediction_entities(
    spec: EvalSpec,
    predictions: list[PredictionRecord],
    extractor: HanLPEntityExtractor,
) -> dict[str, list[Entity]]:
    rows: dict[str, list[Entity]] = {}
    iterator = _progress(predictions, total=len(predictions), desc="Prediction NER", unit="sample")
    for index, prediction in enumerate(iterator, start=1):
        rows[prediction.sample_id] = extractor.extract(prediction.prediction_text)
        if spec.evaluation.progress_every > 0 and index % spec.evaluation.progress_every == 0:
            print(f"Prediction NER {index}/{len(predictions)}")
    return rows


def _summarize_entity_metric(per_sample_rows: list[dict[str, Any]], field_name: str) -> dict[str, float | int]:
    tp = sum(int(row[field_name]["tp"]) for row in per_sample_rows)
    pred_total = sum(int(row[field_name]["pred_total"]) for row in per_sample_rows)
    ref_total = sum(int(row[field_name]["ref_total"]) for row in per_sample_rows)
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / ref_total if ref_total else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "pred_total": pred_total,
        "ref_total": ref_total,
    }


def _summarize_lcs_metric(per_sample_rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not per_sample_rows:
        return {"score": 0.0, "score_sum": 0.0, "active_ref_total": 0, "skipped_ref_total": 0}
    total_score = 0.0
    for row in per_sample_rows:
        for match in row.get("entity_lcs_matches", []):
            total_score += float(match.get("score", 0.0))
    active_ref_total = sum(int(row["entity_lcs"]["active_ref_total"]) for row in per_sample_rows)
    skipped_ref_total = sum(int(row["entity_lcs"]["skipped_ref_total"]) for row in per_sample_rows)
    return {
        "score": total_score / active_ref_total if active_ref_total else 0.0,
        "score_sum": total_score,
        "active_ref_total": active_ref_total,
        "skipped_ref_total": skipped_ref_total,
    }


def _summarize_key_metric(per_sample_rows: list[dict[str, Any]]) -> dict[str, float | int]:
    if not per_sample_rows:
        return {"score": 0.0, "score_sum": 0.0, "active_ref_total": 0, "skipped_ref_total": 0}
    total_score = 0.0
    for row in per_sample_rows:
        for match in row.get("entity_key_matches", []):
            total_score += float(match.get("score", 0.0))
    active_ref_total = sum(int(row["entity_key_recall"]["active_ref_total"]) for row in per_sample_rows)
    skipped_ref_total = sum(int(row["entity_key_recall"]["skipped_ref_total"]) for row in per_sample_rows)
    matched_total = sum(int(row["entity_key_recall"].get("matched_total", 0)) for row in per_sample_rows)
    return {
        "score": total_score / active_ref_total if active_ref_total else 0.0,
        "score_sum": total_score,
        "active_ref_total": active_ref_total,
        "skipped_ref_total": skipped_ref_total,
        "matched_total": matched_total,
    }


def _summarize_lcs_by_label(
    per_sample_rows: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, dict[str, float | int | None]]:
    requested_labels = [label.strip().upper() for label in labels if label.strip()]
    totals = {label: {"score_sum": 0.0, "ref_total": 0} for label in requested_labels}
    for row in per_sample_rows:
        for match in row.get("entity_lcs_matches", []):
            entity = match.get("reference_entity", {})
            label = str(entity.get("label", "")).strip().upper()
            if label not in totals:
                continue
            totals[label]["score_sum"] += float(match.get("score", 0.0))
            totals[label]["ref_total"] += 1

    summary: dict[str, dict[str, float | int | None]] = {}
    for label in requested_labels:
        ref_total = int(totals[label]["ref_total"])
        score_sum = float(totals[label]["score_sum"])
        summary[label] = {
            "recall": score_sum / ref_total if ref_total else None,
            "ref_total": ref_total,
            "score_sum": score_sum,
        }
    return summary


def _summarize_key_by_label(
    per_sample_rows: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, dict[str, float | int | None]]:
    requested_labels = [label.strip().upper() for label in labels if label.strip()]
    totals = {label: {"score_sum": 0.0, "ref_total": 0} for label in requested_labels}
    for row in per_sample_rows:
        for match in row.get("entity_key_matches", []):
            entity = match.get("reference_entity", {})
            label = str(entity.get("label", "")).strip().upper()
            if label not in totals:
                continue
            totals[label]["score_sum"] += float(match.get("score", 0.0))
            totals[label]["ref_total"] += 1

    summary: dict[str, dict[str, float | int | None]] = {}
    for label in requested_labels:
        ref_total = int(totals[label]["ref_total"])
        score_sum = float(totals[label]["score_sum"])
        summary[label] = {
            "recall": score_sum / ref_total if ref_total else None,
            "ref_total": ref_total,
            "score_sum": score_sum,
        }
    return summary


def _lcs_by_label_for_sample(
    lcs_matches: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, dict[str, float | int | None]]:
    requested_labels = [label.strip().upper() for label in labels if label.strip()]
    totals = {label: {"score_sum": 0.0, "ref_total": 0} for label in requested_labels}
    for match in lcs_matches:
        entity = match.get("reference_entity", {})
        label = str(entity.get("label", "")).strip().upper()
        if label not in totals:
            continue
        totals[label]["score_sum"] += float(match.get("score", 0.0))
        totals[label]["ref_total"] += 1

    return {
        label: {
            "recall": float(values["score_sum"]) / int(values["ref_total"]) if int(values["ref_total"]) else None,
            "ref_total": int(values["ref_total"]),
            "score_sum": float(values["score_sum"]),
        }
        for label, values in totals.items()
    }


def _key_by_label_for_sample(
    key_matches: list[dict[str, Any]],
    labels: list[str],
) -> dict[str, dict[str, float | int | None]]:
    requested_labels = [label.strip().upper() for label in labels if label.strip()]
    totals = {label: {"score_sum": 0.0, "ref_total": 0} for label in requested_labels}
    for match in key_matches:
        entity = match.get("reference_entity", {})
        label = str(entity.get("label", "")).strip().upper()
        if label not in totals:
            continue
        totals[label]["score_sum"] += float(match.get("score", 0.0))
        totals[label]["ref_total"] += 1

    return {
        label: {
            "recall": float(values["score_sum"]) / int(values["ref_total"]) if int(values["ref_total"]) else None,
            "ref_total": int(values["ref_total"]),
            "score_sum": float(values["score_sum"]),
        }
        for label, values in totals.items()
    }


def run_eval_spec(spec_path: Path) -> dict[str, Any]:
    spec = load_eval_spec(spec_path)
    ensure_dir(spec.evaluation.output_dir)

    samples = load_samples(spec.dataset)
    predictions = _predict_samples(spec, samples)
    if len(predictions) != len(samples):
        raise ValueError(
            f"Prediction count mismatch: expected {len(samples)}, got {len(predictions)}."
        )

    predictions_by_id = {row.sample_id: row for row in predictions}
    ordered_predictions: list[PredictionRecord] = []
    iterator = _progress(samples, total=len(samples), desc="Entity metrics", unit="sample")
    for sample in iterator:
        prediction = predictions_by_id.get(sample.sample_id)
        if prediction is None:
            raise KeyError(f"Missing prediction for sample id '{sample.sample_id}'.")
        prediction.reference_text = sample.reference_text
        prediction.audio_path = sample.audio_path
        prediction.dataset_prompt = sample.dataset_prompt
        prediction.source_text = sample.source_text
        ordered_predictions.append(prediction)

    prediction_texts = [row.prediction_text for row in ordered_predictions]
    reference_texts = [sample.reference_text for sample in samples]
    bleu, chrf = compute_bleu_chrf(prediction_texts, reference_texts)

    per_sample_rows: list[dict[str, Any]] = []
    reference_entity_rows: list[tuple[str, list[Entity], str, str]] = []
    prediction_entity_rows: list[tuple[str, list[Entity], str, str]] = []
    metrics = set(spec.evaluation.metrics)
    use_entity_lcs = "entity_lcs" in metrics
    use_entity_key_recall = "entity_key_recall" in metrics

    if use_entity_lcs or use_entity_key_recall:
        if not spec.dataset.reference_entity_path or not spec.dataset.reference_entity_path.exists():
            raise ValueError("dataset.reference_entity_path is required for entity_lcs/entity_key_recall evaluation.")
        reference_entities = load_entity_sidecar(spec.dataset.reference_entity_path)
        skip_labels = {label.strip().upper() for label in spec.evaluation.entity_lcs_skip_labels if label.strip()}
        desc = "Entity Key Recall" if use_entity_key_recall else "Entity LCS"
        iterator = _progress(samples, total=len(samples), desc=desc, unit="sample")
        for sample in iterator:
            prediction = predictions_by_id[sample.sample_id]
            ref_entities = reference_entities.get(sample.sample_id, [])
            reference_entity_rows.append((sample.sample_id, ref_entities, sample.reference_text, "reference"))
            if use_entity_key_recall:
                key_counts, key_matches = compute_hard_key_entity_score(
                    prediction.prediction_text,
                    ref_entities,
                    skip_labels=skip_labels,
                )
                row = build_key_metric_row(
                    sample.sample_id,
                    prediction.prediction_text,
                    sample.reference_text,
                    ref_entities,
                    key_counts,
                    key_matches,
                )
                row["entity_key_recall_by_label"] = _key_by_label_for_sample(
                    key_matches,
                    spec.evaluation.entity_lcs_labels,
                )
            else:
                lcs_counts, lcs_matches = compute_lcs_entity_score(
                    prediction.prediction_text,
                    ref_entities,
                    skip_labels=skip_labels,
                )
                row = build_lcs_metric_row(
                    sample.sample_id,
                    prediction.prediction_text,
                    sample.reference_text,
                    ref_entities,
                    lcs_counts,
                    lcs_matches,
                )
                row["entity_lcs_by_label"] = _lcs_by_label_for_sample(
                    lcs_matches,
                    spec.evaluation.entity_lcs_labels,
                )
            per_sample_rows.append(row)
        if use_entity_key_recall:
            entity_key = _summarize_key_metric(per_sample_rows)
            entity_key_by_label = _summarize_key_by_label(per_sample_rows, spec.evaluation.entity_lcs_labels)
            entity_lcs = entity_key
            entity_lcs_by_label = entity_key_by_label
        else:
            entity_lcs = _summarize_lcs_metric(per_sample_rows)
            entity_lcs_by_label = _summarize_lcs_by_label(per_sample_rows, spec.evaluation.entity_lcs_labels)
        entity_f1 = {"precision": 0.0, "recall": entity_lcs["score"], "f1": 0.0, "ref_total": entity_lcs["active_ref_total"], "pred_total": 0}
        entity_soft = {"precision": 0.0, "recall": entity_lcs["score"], "f1": 0.0}
    else:
        extractor = HanLPEntityExtractor(spec.ner)
        reference_entities = _reference_entities(spec, samples, extractor)
        prediction_entities = _prediction_entities(spec, ordered_predictions, extractor)
        embedder = MBertEmbedder(spec.embedding, device=spec.model.device)

        for sample in samples:
            prediction = predictions_by_id[sample.sample_id]
            ref_entities = reference_entities.get(sample.sample_id, [])
            pred_entities = prediction_entities.get(sample.sample_id, [])
            reference_entity_rows.append((sample.sample_id, ref_entities, sample.reference_text, "reference"))
            prediction_entity_rows.append((sample.sample_id, pred_entities, prediction.prediction_text, "prediction"))

            exact_counts = compute_exact_entity_counts(pred_entities, ref_entities)
            entity_texts = [entity.text for entity in pred_entities] + [entity.text for entity in ref_entities]
            if entity_texts:
                embeddings = embedder.encode(entity_texts)
                pred_embeddings = embeddings[: len(pred_entities)]
                ref_embeddings = embeddings[len(pred_entities) :]
                similarity_matrix = cosine_similarity_matrix(pred_embeddings, ref_embeddings)
            else:
                similarity_matrix = []

            soft_counts, soft_matches = compute_soft_entity_counts(
                pred_entities,
                ref_entities,
                similarity_matrix,
                spec.evaluation.entity_soft_tau,
            )
            per_sample_rows.append(
                build_per_sample_metric_row(
                    sample.sample_id,
                    prediction.prediction_text,
                    sample.reference_text,
                    pred_entities,
                    ref_entities,
                    exact_counts,
                    soft_counts,
                    soft_matches,
                )
            )

        entity_f1 = _summarize_entity_metric(per_sample_rows, "entity_f1")
        entity_soft = _summarize_entity_metric(per_sample_rows, "entity_soft")
        entity_lcs = {"score": entity_f1["recall"], "active_ref_total": entity_f1["ref_total"], "skipped_ref_total": 0}
        entity_lcs_by_label = {}

    prediction_rows = [
        {
            "id": row.sample_id,
            "prediction_text": row.prediction_text,
            "reference_text": row.reference_text,
            "audio_path": row.audio_path,
            "dataset_prompt": row.dataset_prompt,
            "source_text": row.source_text,
            "model_name": row.model_name,
            "adapter_path": row.adapter_path,
            "intermediate_asr_text": row.intermediate_asr_text,
            "metadata": row.metadata,
        }
        for row in ordered_predictions
    ]

    summary = {
        "config_path": str(spec_path),
        "dataset_path": str(spec.dataset.path),
        "model_name": spec.model.name,
        "model_kind": spec.model.kind,
        "prediction_mode": spec.evaluation.mode,
        "num_samples": len(samples),
        "bleu": bleu,
        "chrf": chrf,
        "entity_f1_precision": entity_f1["precision"],
        "entity_f1_recall": entity_f1["recall"],
        "entity_f1_f1": entity_f1["f1"],
        "entity_soft_precision": entity_soft["precision"],
        "entity_soft_recall": entity_soft["recall"],
        "entity_soft_f1": entity_soft["f1"],
        "entity_lcs_recall": entity_lcs["score"],
        "entity_recall": entity_lcs["score"],
        "entity_metric_mode": "entity_key_recall" if use_entity_key_recall else ("entity_lcs" if use_entity_lcs else "hanlp_entity_f1"),
        "entity_lcs_averaging": "micro_over_reference_entities" if (use_entity_lcs or use_entity_key_recall) else "not_applicable",
        "num_reference_entities": entity_f1["ref_total"],
        "num_prediction_entities": entity_f1["pred_total"],
        "entity_lcs_score_sum": entity_lcs.get("score_sum", entity_lcs["score"]),
        "num_lcs_reference_entities": entity_lcs["active_ref_total"],
        "num_lcs_skipped_entities": entity_lcs["skipped_ref_total"],
        "entity_lcs_skip_labels": spec.evaluation.entity_lcs_skip_labels,
        "entity_lcs_by_label": entity_lcs_by_label,
        "entity_soft_tau": spec.evaluation.entity_soft_tau,
    }
    if use_entity_key_recall:
        summary["entity_key_recall"] = entity_lcs["score"]
        summary["entity_key_recall_score_sum"] = entity_lcs.get("score_sum", entity_lcs["score"])
        summary["num_key_reference_entities"] = entity_lcs["active_ref_total"]
        summary["num_key_matched_entities"] = entity_lcs.get("matched_total", 0)
    for label, values in entity_lcs_by_label.items():
        if use_entity_key_recall:
            summary[f"entity_key_recall_{label.lower()}"] = values["recall"]
            summary[f"num_key_reference_entities_{label.lower()}"] = values["ref_total"]
        else:
            summary[f"entity_lcs_recall_{label.lower()}"] = values["recall"]
            summary[f"num_lcs_reference_entities_{label.lower()}"] = values["ref_total"]

    write_jsonl(spec.evaluation.output_dir / "predictions.jsonl", prediction_rows)
    write_jsonl(spec.evaluation.output_dir / "metrics.by_sample.jsonl", per_sample_rows)
    write_json(
        spec.evaluation.output_dir / "metrics.summary.json",
        summary,
    )
    write_jsonl(
        spec.evaluation.output_dir / "reference_entities.jsonl",
        entity_records_to_json(reference_entity_rows),
    )
    write_jsonl(
        spec.evaluation.output_dir / "prediction_entities.jsonl",
        entity_records_to_json(prediction_entity_rows),
    )
    write_json(
        spec.evaluation.output_dir / "config.resolved.json",
        spec.raw,
    )
    return summary

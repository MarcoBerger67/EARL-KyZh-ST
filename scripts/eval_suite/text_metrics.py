from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .types import Entity


def get_sacrebleu():
    try:
        import sacrebleu
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError(
            "sacrebleu is required for BLEU/chrF evaluation. Install it with `pip install sacrebleu`."
        ) from exc
    return sacrebleu


def compute_bleu_chrf(predictions: list[str], references: list[str]) -> tuple[float, float]:
    sacrebleu = get_sacrebleu()
    bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="zh").score
    chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=0).score
    return float(bleu), float(chrf)


def normalize_text(text: str) -> str:
    text = str(text).strip()
    if text.startswith("Translation:"):
        text = text[len("Translation:") :].strip()
    return " ".join(text.split())


def normalize_key_text(text: str) -> str:
    return normalize_text(text).replace(" ", "").lower()


def longest_common_substring_len(left: str, right: str) -> int:
    if not left or not right:
        return 0
    best = 0
    previous = [0] * (len(right) + 1)
    for char in left:
        current = [0] * (len(right) + 1)
        for index, other in enumerate(right):
            if char == other:
                current[index + 1] = previous[index] + 1
                if current[index + 1] > best:
                    best = current[index + 1]
        previous = current
    return best


def _micro_summary(tp: int, pred_total: int, ref_total: int) -> dict[str, float | int]:
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / ref_total if ref_total else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "tp": tp,
        "pred_total": pred_total,
        "ref_total": ref_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_exact_entity_counts(predicted: list[Entity], reference: list[Entity]) -> dict[str, float | int]:
    matched_ref = [False] * len(reference)
    tp = 0
    for entity in predicted:
        for index, ref_entity in enumerate(reference):
            if matched_ref[index]:
                continue
            if entity.label == ref_entity.label and entity.text == ref_entity.text:
                matched_ref[index] = True
                tp += 1
                break
    return _micro_summary(tp, len(predicted), len(reference))


def compute_lcs_entity_score(
    prediction_text: str,
    reference: list[Entity],
    skip_labels: set[str] | None = None,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    if skip_labels is None:
        skip_labels = {"PER"}
    active_refs = [
        entity
        for entity in reference
        if str(entity.label).strip().upper() not in skip_labels
    ]
    if not active_refs:
        return {
            "score": 1.0,
            "ref_total": 0,
            "active_ref_total": 0,
            "skipped_ref_total": len(reference),
        }, []

    normalized_prediction = normalize_key_text(prediction_text)
    rows: list[dict[str, Any]] = []
    total = 0.0
    for index, entity in enumerate(active_refs):
        normalized_entity = normalize_key_text(entity.text)
        if normalized_entity:
            lcs = longest_common_substring_len(normalized_entity, normalized_prediction)
            ratio = lcs / len(normalized_entity)
        else:
            lcs = 0
            ratio = 0.0
        total += ratio
        rows.append(
            {
                "reference_index": index,
                "reference_entity": asdict(entity),
                "normalized_reference": normalized_entity,
                "lcs_length": lcs,
                "reference_length": len(normalized_entity),
                "score": ratio,
            }
        )

    score = total / len(active_refs)
    return {
        "score": score,
        "ref_total": len(reference),
        "active_ref_total": len(active_refs),
        "skipped_ref_total": len(reference) - len(active_refs),
    }, rows


def compute_hard_key_entity_score(
    prediction_text: str,
    reference: list[Entity],
    skip_labels: set[str] | None = None,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    """Hard key recall over reference entities.

    Each reference entity gets 1 only if its normalized full text appears as a
    contiguous substring in the normalized prediction; otherwise it gets 0.
    This intentionally does not give LCS partial credit.
    """
    if skip_labels is None:
        skip_labels = {"PER"}
    active_refs = [
        entity
        for entity in reference
        if str(entity.label).strip().upper() not in skip_labels
    ]
    if not active_refs:
        return {
            "score": 1.0,
            "ref_total": 0,
            "active_ref_total": 0,
            "skipped_ref_total": len(reference),
        }, []

    normalized_prediction = normalize_key_text(prediction_text)
    rows: list[dict[str, Any]] = []
    matched = 0
    for index, entity in enumerate(active_refs):
        normalized_entity = normalize_key_text(entity.text)
        is_match = bool(normalized_entity and normalized_entity in normalized_prediction)
        score = 1.0 if is_match else 0.0
        matched += int(is_match)
        rows.append(
            {
                "reference_index": index,
                "reference_entity": asdict(entity),
                "normalized_reference": normalized_entity,
                "matched": is_match,
                "score": score,
            }
        )

    score = matched / len(active_refs)
    return {
        "score": score,
        "ref_total": len(reference),
        "active_ref_total": len(active_refs),
        "skipped_ref_total": len(reference) - len(active_refs),
        "matched_total": matched,
    }, rows


def compute_soft_entity_counts(
    predicted: list[Entity],
    reference: list[Entity],
    similarity_matrix: list[list[float]],
    tau: float,
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    from .entity_backend import soft_entity_matching

    matches = soft_entity_matching(predicted, reference, similarity_matrix, tau)
    tp = len(matches)
    match_rows = [
        {
            "prediction_index": left,
            "reference_index": right,
            "similarity": score,
            "prediction_entity": asdict(predicted[left]),
            "reference_entity": asdict(reference[right]),
        }
        for left, right, score in matches
    ]
    return _micro_summary(tp, len(predicted), len(reference)), match_rows


def build_per_sample_metric_row(
    sample_id: str,
    prediction_text: str,
    reference_text: str,
    predicted_entities: list[Entity],
    reference_entities: list[Entity],
    exact_counts: dict[str, float | int],
    soft_counts: dict[str, float | int],
    soft_matches: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "prediction_text": prediction_text,
        "reference_text": reference_text,
        "prediction_entities": [asdict(entity) for entity in predicted_entities],
        "reference_entities": [asdict(entity) for entity in reference_entities],
        "entity_f1": exact_counts,
        "entity_soft": soft_counts,
        "entity_soft_matches": soft_matches,
    }


def build_lcs_metric_row(
    sample_id: str,
    prediction_text: str,
    reference_text: str,
    reference_entities: list[Entity],
    lcs_counts: dict[str, float | int],
    lcs_matches: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "prediction_text": prediction_text,
        "reference_text": reference_text,
        "prediction_entities": [],
        "reference_entities": [asdict(entity) for entity in reference_entities],
        "entity_lcs": lcs_counts,
        "entity_lcs_matches": lcs_matches,
    }


def build_key_metric_row(
    sample_id: str,
    prediction_text: str,
    reference_text: str,
    reference_entities: list[Entity],
    key_counts: dict[str, float | int],
    key_matches: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": sample_id,
        "prediction_text": prediction_text,
        "reference_text": reference_text,
        "reference_entities": [asdict(entity) for entity in reference_entities],
        "entity_key_recall": key_counts,
        "entity_key_matches": key_matches,
    }

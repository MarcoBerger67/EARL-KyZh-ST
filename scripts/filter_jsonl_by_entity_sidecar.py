from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


VALID_LABELS = {"PER", "LOC", "ORG", "TIME", "NUM", "TERM", "TITLE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter a training JSONL and its entity sidecar to records whose sidecar "
            "contains at least one valid entity."
        )
    )
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--entity-path", type=Path, required=True)
    parser.add_argument("--output-data-path", type=Path, required=True)
    parser.add_argument("--output-entity-path", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument(
        "--allow-labels",
        type=str,
        default=",".join(sorted(VALID_LABELS)),
        help="Comma-separated valid labels. Empty value disables label filtering.",
    )
    parser.add_argument(
        "--drop-invalid-entities",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop malformed entities and entities with labels outside --allow-labels.",
    )
    parser.add_argument(
        "--strict-id-coverage",
        action="store_true",
        help="Fail when data ids and sidecar ids are not exactly the same set.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object in {path} line {line_number}.")
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def get_record_id(row: dict[str, Any]) -> str | None:
    value = row.get("id") or row.get("key")
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def normalize_entity(item: Any, valid_labels: set[str] | None) -> dict[str, str] | None:
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
        return None
    if valid_labels is not None and label not in valid_labels:
        return None
    return {"text": text, "label": label}


def normalize_entities(
    row: dict[str, Any],
    valid_labels: set[str] | None,
    drop_invalid_entities: bool,
) -> tuple[list[dict[str, str]], int]:
    raw_entities = row.get("entities")
    if not isinstance(raw_entities, list):
        return [], 0

    entities: list[dict[str, str]] = []
    invalid_count = 0
    seen: set[tuple[str, str]] = set()
    for item in raw_entities:
        entity = normalize_entity(item, valid_labels)
        if entity is None:
            invalid_count += 1
            if drop_invalid_entities:
                continue
            entity = normalize_entity(item, None)
            if entity is None:
                continue
        key = (entity["text"], entity["label"])
        if key in seen:
            continue
        seen.add(key)
        entities.append(entity)
    return entities, invalid_count


def strip_internal_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(strip_internal_fields(row), ensure_ascii=False) + "\n")


def duplicate_ids(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(record_id for row in rows if (record_id := get_record_id(row)))
    return {record_id: count for record_id, count in counts.items() if count > 1}


def main() -> None:
    args = parse_args()
    valid_labels = None
    if args.allow_labels.strip():
        valid_labels = {label.strip().upper() for label in args.allow_labels.split(",") if label.strip()}

    data_rows = read_jsonl(args.data_path)
    entity_rows = read_jsonl(args.entity_path)
    data_duplicate_ids = duplicate_ids(data_rows)
    entity_duplicate_ids = duplicate_ids(entity_rows)
    if data_duplicate_ids:
        raise ValueError(f"Duplicate ids in data file: {list(data_duplicate_ids.items())[:10]}")
    if entity_duplicate_ids:
        raise ValueError(f"Duplicate ids in entity sidecar: {list(entity_duplicate_ids.items())[:10]}")

    data_by_id: dict[str, dict[str, Any]] = {}
    missing_data_ids = 0
    for row in data_rows:
        record_id = get_record_id(row)
        if record_id is None:
            missing_data_ids += 1
            continue
        data_by_id[record_id] = row

    entity_by_id: dict[str, dict[str, Any]] = {}
    malformed_entity_rows = 0
    invalid_entity_items = 0
    empty_entity_rows = 0
    label_counts: Counter[str] = Counter()
    for row in entity_rows:
        record_id = get_record_id(row)
        if record_id is None:
            malformed_entity_rows += 1
            continue
        entities, invalid_count = normalize_entities(
            row,
            valid_labels=valid_labels,
            drop_invalid_entities=args.drop_invalid_entities,
        )
        invalid_entity_items += invalid_count
        if not isinstance(row.get("entities"), list):
            malformed_entity_rows += 1
        if not entities:
            empty_entity_rows += 1
        for entity in entities:
            label_counts[entity["label"]] += 1
        entity_by_id[record_id] = {"id": record_id, "entities": entities}

    data_ids = set(data_by_id)
    entity_ids = set(entity_by_id)
    missing_sidecar_ids = sorted(data_ids - entity_ids)
    extra_sidecar_ids = sorted(entity_ids - data_ids)
    if args.strict_id_coverage and (missing_sidecar_ids or extra_sidecar_ids):
        raise ValueError(
            "Data ids and sidecar ids do not match exactly: "
            f"missing_sidecar={len(missing_sidecar_ids)}, extra_sidecar={len(extra_sidecar_ids)}"
        )

    output_data_rows: list[dict[str, Any]] = []
    output_entity_rows: list[dict[str, Any]] = []
    for data_row in data_rows:
        record_id = get_record_id(data_row)
        if record_id is None:
            continue
        sidecar_row = entity_by_id.get(record_id)
        if not sidecar_row or not sidecar_row["entities"]:
            continue
        output_data_rows.append(data_row)
        output_entity_rows.append(sidecar_row)

    write_jsonl(args.output_data_path, output_data_rows)
    write_jsonl(args.output_entity_path, output_entity_rows)

    report = {
        "data_path": str(args.data_path),
        "entity_path": str(args.entity_path),
        "output_data_path": str(args.output_data_path),
        "output_entity_path": str(args.output_entity_path),
        "input_data_rows": len(data_rows),
        "input_entity_rows": len(entity_rows),
        "missing_data_ids": missing_data_ids,
        "malformed_entity_rows": malformed_entity_rows,
        "invalid_or_dropped_entity_items": invalid_entity_items,
        "empty_entity_rows": empty_entity_rows,
        "missing_sidecar_ids": len(missing_sidecar_ids),
        "extra_sidecar_ids": len(extra_sidecar_ids),
        "kept_rows": len(output_data_rows),
        "dropped_rows": len(data_rows) - len(output_data_rows),
        "kept_entity_items": sum(len(row["entities"]) for row in output_entity_rows),
        "label_counts": dict(sorted(label_counts.items())),
        "strict_id_coverage": bool(args.strict_id_coverage),
        "format_ok_for_group_risk": (
            not missing_data_ids
            and not data_duplicate_ids
            and not entity_duplicate_ids
            and not malformed_entity_rows
            and not missing_sidecar_ids
        ),
    }
    report_path = args.report_path or args.output_data_path.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()

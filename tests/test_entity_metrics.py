from eval_suite.text_metrics import compute_hard_key_entity_score, normalize_key_text
from eval_suite.types import Entity


def test_normalize_key_text_removes_spaces_and_case():
    assert normalize_key_text(" Beijing University ") == "beijinguniversity"


def test_hard_key_recall_requires_full_entity_match():
    reference = [
        Entity(text="北京大学", label="ORG"),
        Entity(text="北京", label="LOC"),
        Entity(text="艾达尔", label="PER"),
    ]
    counts, matches = compute_hard_key_entity_score(
        prediction_text="北京大学位于北京。",
        reference=reference,
        skip_labels=[],
    )

    assert counts["active_ref_total"] == 3
    assert counts["matched_total"] == 2
    assert counts["score"] == 2 / 3
    assert [row["matched"] for row in matches] == [True, True, False]


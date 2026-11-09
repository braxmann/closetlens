import pandas as pd
import pytest

from wardrobe.evaluation.metrics import NULL, confusion, join, summary, wilson_interval


def make_joined():
    test_set = pd.DataFrame({
        "attribute": ["category", "category", "category", "formality", "formality"],
        "id": [1, 2, 3, 1, 2],
        "label": ["shirt", "shirt", "pants", "casual", "formal"],
    })
    predictions = pd.DataFrame({
        "id": [1, 2, 3, 99],
        "seconds": [1.0, 1.0, 1.0, 1.0],
        "error": [None, None, None, None],
        "category": ["shirt", "other", "pants", "dress"],
        "formality": ["casual", None, "casual", "casual"],
    })
    return join(test_set, predictions)


def test_null_answers_lower_coverage_not_accuracy():
    result = summary(make_joined()).loc["formality"]
    assert result["coverage"] == 0.5
    assert result["accuracy"] == 1.0


def test_answer_outside_true_classes_is_an_error():
    result = summary(make_joined()).loc["category"]
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_answers_for_images_outside_the_test_set_are_ignored():
    assert 99 not in make_joined()["id"].values


def test_confusion_shows_null_and_foreign_answers():
    assert NULL in confusion(make_joined(), "formality").columns
    assert "other" in confusion(make_joined(), "category").columns


def test_wilson_interval_stays_inside_zero_and_one():
    low, high = wilson_interval(10, 10)
    assert 0.6 < low < 1.0
    assert high == pytest.approx(1.0)

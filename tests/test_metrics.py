"""Tests de métricas."""

import pytest

from alpr.eval.metrics import (
    average_levenshtein,
    character_accuracy_rate,
    detection_metrics,
    exact_plate_match,
    iou,
    levenshtein,
    word_accuracy,
)


def test_iou_identical() -> None:
    box = (0, 0, 10, 10)
    assert iou(box, box) == pytest.approx(1.0)


def test_iou_no_overlap() -> None:
    a = (0, 0, 10, 10)
    b = (20, 20, 30, 30)
    assert iou(a, b) == pytest.approx(0.0)


def test_levenshtein() -> None:
    assert levenshtein("ABC1D23", "ABC1D23") == 0
    assert levenshtein("ABC1D23", "ABC1D24") == 1
    assert levenshtein("ABC", "ABCD") == 1


def test_character_accuracy_rate() -> None:
    assert character_accuracy_rate(["ABC1D23"], ["ABC1D23"]) == 1.0
    assert character_accuracy_rate(["ABC1D23"], ["ABC1D24"]) == pytest.approx(6 / 7)


def test_word_accuracy() -> None:
    assert word_accuracy(["ABC1D23", "AB123CD"], ["ABC1D23", "AB123CD"]) == 1.0
    assert word_accuracy(["ABC1D23", "AB123CD"], ["ABC1D23", "AB123CE"]) == 0.5


def test_exact_plate_match() -> None:
    assert exact_plate_match(["ABC1D23"], ["ABC1D23"]) == 1.0


def test_average_levenshtein() -> None:
    assert average_levenshtein(["ABC1D23", "AB123CD"], ["ABC1D23", "AB123CE"]) == 0.5


def test_detection_metrics() -> None:
    pred = [[(0, 0, 10, 10), (20, 20, 30, 30)]]
    gt = [[(0, 0, 10, 10)]]
    m = detection_metrics(pred, gt)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(1.0)

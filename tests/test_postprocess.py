"""Tests de postprocesamiento."""

import pytest

from alpr.core.postprocess import clean_plate_text, is_mercosur_format, post_process


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("abc-1d23", "ABC1D23"),
        ("AB 123 CD", "AB123CD"),
        ("!!!abc1234???", "ABC1234"),
    ],
)
def test_clean_plate_text(raw: str, expected: str) -> None:
    assert clean_plate_text(raw) == expected


@pytest.mark.parametrize(
    "text,valid",
    [
        ("ABC1D23", True),
        ("AB123CD", True),
        ("ABCD123", True),
        ("ABC1234", True),
        ("1234567", False),
        ("ABCDEFG", False),
    ],
)
def test_is_mercosur_format(text: str, valid: bool) -> None:
    assert is_mercosur_format(text) is valid


def test_post_process() -> None:
    result = post_process("a-b-c-1-d-2-3", 0.87)
    assert result["text"] == "ABC1D23"
    assert result["valid_format"] is True
    assert result["confidence"] == pytest.approx(0.87)

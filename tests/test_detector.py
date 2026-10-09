"""Tests for the score-to-label mapping. These do not load CLIP or any weights."""

import pytest

from src.detector import label_from_probability


@pytest.mark.parametrize(
    "p, expected",
    [
        (0.99, "ai"),
        (0.65, "ai"),       # boundary: high is inclusive for "ai"
        (0.50, "uncertain"),
        (0.35, "real"),     # boundary: low is inclusive for "real"
        (0.01, "real"),
    ],
)
def test_label_from_probability(p, expected):
    assert label_from_probability(p, low=0.35, high=0.65) == expected


@pytest.mark.parametrize("p", [-0.1, 1.2])
def test_rejects_out_of_range(p):
    with pytest.raises(ValueError):
        label_from_probability(p)

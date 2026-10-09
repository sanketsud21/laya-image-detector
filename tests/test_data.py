"""Tests for splitting logic. These do not download any model."""

import pandas as pd
import pytest

from src.data import split_holdout_generator, split_random


def make_df() -> pd.DataFrame:
    rows = [{"path": f"real_{i}", "label": 0, "generator": "real"} for i in range(50)]
    for gen in ["sd15", "sdxl", "flux"]:
        rows += [{"path": f"{gen}_{i}", "label": 1, "generator": gen} for i in range(30)]
    return pd.DataFrame(rows).reset_index(drop=True)


def test_holdout_generator_absent_from_train():
    df = make_df()
    train, _ = split_holdout_generator(df, "sdxl", seed=0)
    assert "sdxl" not in set(train["generator"])


def test_holdout_test_contains_only_that_generator_plus_real():
    df = make_df()
    _, test = split_holdout_generator(df, "sdxl", seed=0)
    assert set(test["generator"]) == {"sdxl", "real"}


def test_holdout_test_has_both_classes():
    df = make_df()
    _, test = split_holdout_generator(df, "sdxl", seed=0)
    assert set(test["label"]) == {0, 1}


def test_holdout_split_is_disjoint_and_complete():
    df = make_df()
    train, test = split_holdout_generator(df, "flux", seed=0)
    assert set(train.index).isdisjoint(test.index)
    assert len(train) + len(test) == len(df)


def test_holdout_rejects_real_and_unknown():
    df = make_df()
    with pytest.raises(ValueError):
        split_holdout_generator(df, "real")
    with pytest.raises(ValueError):
        split_holdout_generator(df, "midjourney")


def test_random_split_is_stratified_and_disjoint():
    df = make_df()
    train, test = split_random(df, test_frac=0.2, seed=0)
    assert set(train.index).isdisjoint(test.index)
    assert abs(test["label"].mean() - df["label"].mean()) < 0.05

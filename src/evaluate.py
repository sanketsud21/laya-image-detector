"""Metrics for the detector. Everything takes plain arrays, so it is easy to test.

Two things to keep in mind when reading these numbers:
- Accuracy hides the error that matters most. A false positive means a real photo
  gets labelled as AI, so report the false positive rate alongside accuracy.
- Scores are not automatically probabilities. calibration_table checks whether a
  score of 0.8 is right about 80% of the time.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    """Summary metrics at one threshold. Values are plain Python types for JSON."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    has_both_classes = len(np.unique(y_true)) == 2

    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score)) if has_both_classes else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def per_generator_recall(df: pd.DataFrame, y_score, threshold: float = 0.5) -> pd.Series:
    """Share of each AI generator's images flagged as AI.

    An aggregate score can hide one generator being missed entirely. This breaks it out.
    df must be row-aligned with y_score.
    """
    flagged = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    d = df.assign(flagged=flagged)
    return d[d["label"] == 1].groupby("generator")["flagged"].mean()


def threshold_sweep(y_true, y_score, thresholds=None) -> pd.DataFrame:
    """Precision, recall, and false positive rate across thresholds, for choosing a cutoff (M4)."""
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        m = compute_metrics(y_true, y_score, threshold=float(t))
        rows.append(
            {
                "threshold": round(float(t), 3),
                "precision": m["precision"],
                "recall": m["recall"],
                "false_positive_rate": m["false_positive_rate"],
            }
        )
    return pd.DataFrame(rows)


def calibration_table(y_true, y_score, n_bins: int = 10) -> pd.DataFrame:
    """Compare mean predicted score with the observed fraction of AI images in each bin.

    A well-calibrated model has points close to the diagonal. Empty bins are dropped.
    """
    frac_pos, mean_pred = calibration_curve(
        np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float), n_bins=n_bins, strategy="uniform"
    )
    return pd.DataFrame({"mean_predicted": mean_pred, "fraction_ai": frac_pos})

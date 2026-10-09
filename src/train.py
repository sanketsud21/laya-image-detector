"""Train a linear probe on CLIP embeddings and report metrics.

Examples:
    python -m src.train                    # random split (M2 baseline)
    python -m src.train --holdout sdxl     # leave one generator out (M3)

Outputs:
    models/detector.joblib     trained classifier + metadata needed for inference
    reports/metrics.json       metrics on the test split
"""

import argparse
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression

from src.data import load_manifest, split_holdout_generator, split_random
from src.evaluate import calibration_table, compute_metrics, per_generator_recall, threshold_sweep
from src.features import DEFAULT_MODEL_ID, ClipEncoder, load_or_embed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CLIP linear probe.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--cache", type=Path, default=Path("data/processed/embeddings.npz"))
    parser.add_argument("--holdout", default=None, help="AI generator to hold out entirely")
    parser.add_argument("--model-out", type=Path, default=Path("models/detector.joblib"))
    parser.add_argument("--metrics-out", type=Path, default=Path("reports/metrics.json"))
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = load_manifest(args.manifest)

    if args.holdout:
        mode = f"holdout:{args.holdout}"
        train_df, test_df = split_holdout_generator(df, args.holdout, seed=args.seed)
    else:
        mode = "random"
        train_df, test_df = split_random(df, seed=args.seed)

    print(f"Mode: {mode} | train={len(train_df)} test={len(test_df)}")

    encoder = ClipEncoder(DEFAULT_MODEL_ID)
    X = load_or_embed(df, encoder, args.cache)

    X_train = X[train_df.index.to_numpy()]
    y_train = train_df["label"].to_numpy()
    X_test = X[test_df.index.to_numpy()]
    y_test = test_df["label"].to_numpy()

    # class_weight="balanced" compensates for uneven real/AI counts.
    clf = LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "mode": mode,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "threshold": args.threshold,
        "overall": compute_metrics(y_test, scores, args.threshold),
        "per_generator_recall": per_generator_recall(test_df, scores, args.threshold).to_dict(),
        "threshold_sweep": threshold_sweep(y_test, scores).to_dict(orient="list"),
        "calibration": calibration_table(y_test, scores).to_dict(orient="list"),
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"clf": clf, "model_id": encoder.model_id, "version": f"clip-linear-probe/{mode}/C={args.C}"},
        args.model_out,
    )

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    o = metrics["overall"]
    print(
        f"accuracy={o['accuracy']:.3f}  precision={o['precision']:.3f}  "
        f"recall={o['recall']:.3f}  FPR={o['false_positive_rate']:.3f}  "
        f"AUC={o['roc_auc'] if o['roc_auc'] is None else round(o['roc_auc'], 3)}"
    )
    print("Per-generator recall:", {k: round(v, 3) for k, v in metrics["per_generator_recall"].items()})
    print(f"Saved model to {args.model_out} and metrics to {args.metrics_out}")


if __name__ == "__main__":
    main()

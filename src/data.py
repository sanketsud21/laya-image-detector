"""Build a manifest of labelled images and split it for training and evaluation.

Expected raw layout. Each folder is one source, and the folder name is the generator:

    data/raw/
        real/     # real photographs            -> label 0
        sd15/     # Stable Diffusion 1.5 outputs -> label 1
        sdxl/     # any other generator          -> label 1

The manifest (CSV with path, label, generator) is the single source of truth for
every later step, so splits are reproducible and easy to inspect.

Usage:
    python -m src.data --raw data/raw --out data/processed/manifest.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REAL_FOLDER = "real"
REAL, AI = 0, 1


def build_manifest(raw_dir: Path, out_path: Path) -> pd.DataFrame:
    """Scan raw_dir and write one row per image: path, label, generator."""
    rows = []
    for folder in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        is_real = folder.name.lower() == REAL_FOLDER
        label = REAL if is_real else AI
        generator = "real" if is_real else folder.name
        for f in sorted(folder.rglob("*")):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS:
                rows.append({"path": str(f.resolve()), "label": label, "generator": generator})

    if not rows:
        raise ValueError(f"No images found under {raw_dir}")

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def load_manifest(path: Path) -> pd.DataFrame:
    """Read a manifest and reset the index so row positions match embedding rows."""
    return pd.read_csv(path).reset_index(drop=True)


def load_rgb(path) -> Image.Image:
    """Open an image and convert to RGB. Some sources use RGBA or grayscale."""
    with Image.open(path) as img:
        return img.convert("RGB")


def split_random(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0):
    """Stratified random split. Good for a baseline, but it hides generator shift (M2)."""
    train, test = train_test_split(
        df, test_size=test_frac, stratify=df["label"], random_state=seed
    )
    return train, test


def split_holdout_generator(df: pd.DataFrame, holdout: str, real_test_frac: float = 0.2, seed: int = 0):
    """Hold out one AI generator entirely, plus a random slice of real images.

    The held-out generator never appears in training, so the test score measures
    generalization to an unseen generator (M3). Real images are split by a random
    fraction so the test set contains both classes.
    """
    generators = sorted(set(df["generator"]))
    if holdout == "real" or holdout not in generators:
        raise ValueError(f"Choose an AI generator to hold out. Options: {[g for g in generators if g != 'real']}")

    fake_test = df[df["generator"] == holdout]
    real_test = df[df["label"] == REAL].sample(frac=real_test_frac, random_state=seed)
    test = pd.concat([fake_test, real_test])
    train = df.drop(index=test.index)
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the image manifest.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/manifest.csv"))
    args = parser.parse_args()

    df = build_manifest(args.raw, args.out)
    print(df.groupby(["generator", "label"]).size().to_string())
    print(f"\nWrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()

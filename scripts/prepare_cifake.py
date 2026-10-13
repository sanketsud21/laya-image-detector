"""Merge CIFAKE's train/test splits into the layout src/data.py expects.

data/{train,test}/{REAL,FAKE}  ->  data/raw/{real,sd}/
"""

import shutil
import sys
from pathlib import Path

SRC = Path("data")
DST = Path("data/raw")
SPLITS = ["train", "test"]
CLASS_MAP = {"REAL": "real", "FAKE": "sd"}  # "sd" = Stable Diffusion generator


def main() -> None:
    if DST.exists() and any(DST.iterdir()):
        sys.exit(f"{DST} already contains files. Delete it first to avoid duplicates.")

    for split in SPLITS:
        for src_class, dst_class in CLASS_MAP.items():
            src_dir = SRC / split / src_class
            if not src_dir.is_dir():
                sys.exit(f"Missing folder: {src_dir}")

            dst_dir = DST / dst_class
            dst_dir.mkdir(parents=True, exist_ok=True)

            files = [f for f in src_dir.iterdir() if f.is_file()]
            for f in files:
                # Prefix with the split so train and test filenames can't collide
                shutil.copy2(f, dst_dir / f"{split}_{f.name}")
            print(f"{split}/{src_class} -> {dst_dir}: {len(files)} files")


if __name__ == "__main__":
    main()
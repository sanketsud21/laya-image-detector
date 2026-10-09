"""Turn images into fixed-length vectors with a frozen CLIP encoder.

The encoder is never trained here. It is a feature extractor, and the classifier
learns on top of its vectors. That keeps the demo cheap enough to run on a laptop.

Note on preprocessing: CLIP's processor resizes the short side to 224 and
center-crops to 224x224. Anything outside that crop is discarded before the model
sees it. This is one of the lessons in docs/learning-log.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.data import load_rgb

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"


class ClipEncoder:
    """Wraps a pretrained CLIP image tower and returns L2-normalized embeddings."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        """Return an array of shape (len(images), embedding_dim)."""
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self.model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.cpu().numpy()


def embed_paths(encoder: ClipEncoder, paths: list[str], batch_size: int = 32) -> np.ndarray:
    """Load images in batches and embed them. Batching bounds memory use."""
    chunks = []
    for start in range(0, len(paths), batch_size):
        batch = [load_rgb(p) for p in paths[start : start + batch_size]]
        chunks.append(encoder.encode(batch))
    return np.concatenate(chunks)


def load_or_embed(df: pd.DataFrame, encoder: ClipEncoder, cache_path: Path, batch_size: int = 32) -> np.ndarray:
    """Embed every row of df, reusing the cache only if paths and model match exactly."""
    paths = df["path"].to_numpy().astype(str)

    if cache_path.exists():
        cached = np.load(cache_path)
        same_paths = np.array_equal(cached["paths"], paths)
        same_model = str(cached["model_id"]) == encoder.model_id
        if same_paths and same_model:
            print(f"Loaded cached embeddings from {cache_path}")
            return cached["X"]
        print("Cache is stale (manifest or model changed); recomputing.")

    print(f"Embedding {len(paths)} images with {encoder.model_id} ...")
    X = embed_paths(encoder, list(paths), batch_size=batch_size)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, X=X, paths=paths, model_id=np.array(encoder.model_id))
    return X

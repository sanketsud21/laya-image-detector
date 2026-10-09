"""Score an image and turn the score into a label.

This module is the seam between the image model and Laya's decision layer.

The shapes of ImageScorer and DetectionResult are placeholders until milestone M1,
when we know how Laya's text model feeds its decision logic. Once that is clear,
adapt these types to match Laya's interface, not the other way round.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import joblib
from PIL import Image

from src.features import ClipEncoder

Label = Literal["ai", "real", "uncertain"]


class ImageScorer(Protocol):
    """Anything that maps an image to a probability that it is AI-generated."""

    def score(self, image: Image.Image) -> float: ...


def label_from_probability(p: float, low: float = 0.35, high: float = 0.65) -> Label:
    """Map a probability to a label. Scores between low and high are 'uncertain'.

    The uncertain band is a policy choice. It trades coverage (how many images get a
    firm answer) against the risk of confidently wrong labels. The default values
    are not tuned; M4 is where they should be chosen.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Probability must be in [0, 1], got {p}")
    if p >= high:
        return "ai"
    if p <= low:
        return "real"
    return "uncertain"


@dataclass(frozen=True)
class DetectionResult:
    ai_probability: float
    label: Label
    model_version: str

    def to_dict(self) -> dict:
        return asdict(self)


class ImageDetector:
    """Loads a trained linear probe and runs it on new images."""

    def __init__(
        self,
        model_path: Path,
        encoder: ClipEncoder | None = None,
        low: float = 0.35,
        high: float = 0.65,
    ):
        bundle = joblib.load(model_path)
        self.clf = bundle["clf"]
        self.version: str = bundle["version"]
        self.encoder = encoder or ClipEncoder(bundle["model_id"])
        self.low = low
        self.high = high

    def score(self, image: Image.Image) -> float:
        feats = self.encoder.encode([image.convert("RGB")])
        return float(self.clf.predict_proba(feats)[0, 1])

    def predict(self, image: Image.Image) -> DetectionResult:
        p = self.score(image)
        return DetectionResult(
            ai_probability=round(p, 4),
            label=label_from_probability(p, self.low, self.high),
            model_version=self.version,
        )

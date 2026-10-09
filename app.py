"""Gradio demo for the image detector.

Run with:
    python app.py
"""

from pathlib import Path

import gradio as gr

from src.detector import ImageDetector

MODEL_PATH = Path("models/detector.joblib")

DISCLAIMER = (
    "Educational demo. The score reflects patterns learned from a small training set. "
    "It is not proof that an image is or is not AI-generated, and it can be wrong, "
    "especially on compressed, edited, or unfamiliar images."
)

detector = ImageDetector(MODEL_PATH) if MODEL_PATH.exists() else None


def predict(image):
    if detector is None:
        return {"error": "No trained model found. Run `python -m src.train` first."}
    if image is None:
        return {"error": "Upload an image to classify."}
    return detector.predict(image).to_dict()


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Image"),
    outputs=gr.JSON(label="Result"),
    title="Laya image detector (demo)",
    description=DISCLAIMER,
)

if __name__ == "__main__":
    demo.launch()

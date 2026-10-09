# Laya: how the current text model works

**Status:** Not yet documented. This file is the M1 deliverable and needs the Laya source to fill in.

The goal is to answer one question: where does an image score enter Laya's decision layer, and what shape must it have?

## 1. Inputs

- How is text passed to the model? (Tokenization, maximum length, batching)
- What metadata accompanies the text, if any?
- Where does input validation happen?

## 2. The model

- Framework and model architecture
- Where are the weights loaded from, and what size are they?
- What is the output? (Logits, probabilities, class labels, or free text?)
- Is the model called synchronously per request, or in batches?

## 3. Decision layer

- What consumes the model's output?
- Are there thresholds, rules, or post-processing steps?
- What does the final decision look like? (Label, score, reason, action)
- Is there a fallback when the model is unsure or fails?

## 4. Integration points for an image score

- Can the decision layer accept a new signal such as `image_ai_probability`?
- Does it expect a single score, or a structured result?
- Where should the image branch run? (Same service or a separate one)

## 5. Open questions

- [ ] Which file or module is the entry point for classification?
- [ ] What does the decision layer return to the caller?
- [ ] Are there existing tests or example inputs I can reuse?
- [ ] Which license applies to Laya, and does it allow this demo to reference it?

## 6. Mapping to `src/detector.py`

Once the sections above are filled in, adjust `ImageScorer` and `DetectionResult` in `src/detector.py` so they match Laya's interface. Record the decision here, with the reason.

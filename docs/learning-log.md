# Learning log

Each milestone ends with a short entry: what you did, what surprised you, and what you would do differently. The entries are the real deliverable of this project. The code is the evidence.

Keep entries short and concrete. Include numbers where you have them.

---

## M0: Images as data

**Goal:** Understand what an image looks like to a program.

- [ ] Load a few images and print their size, mode (RGB, RGBA, L), and dtype
- [ ] Check whether real and AI images differ in resolution, format, or file size
- [ ] Look at the same image before and after CLIP's preprocessing

**Lesson:** _(fill in)_

---

## M1: Understand Laya

**Goal:** Document how Laya's text model feeds its decision layer. See `docs/laya-notes.md`.

**Lesson:** _(fill in)_

---

## M2: Baseline

**Goal:** A CLIP embedding and logistic regression baseline on a random split.

- Random-split accuracy: _(fill in)_
- Random-split false positive rate: _(fill in)_

**Lesson:** _(fill in). What does the number tell you, and what does it not tell you?_

---

## M3: Honest evaluation

**Goal:** Hold out one generator entirely and compare with M2.

| Split | Accuracy | Recall | False positive rate | AUC |
|---|---|---|---|---|
| Random | | | | |
| Holdout: _(generator)_ | | | | |

**Lesson:** _(fill in). This drop is usually the most important finding in the project._

---

## M4: Calibration and thresholds

**Goal:** Check whether scores mean what they say, and choose a threshold with an explicit trade-off.

- Calibration: does a score of 0.8 correspond to about 80% AI?
- Chosen threshold and why: _(fill in)_
- False positive rate at that threshold: _(fill in)_

**Lesson:** _(fill in)_

---

## M5: Interface and demo

**Goal:** Expose the detector behind a clean interface and a small UI.

**Lesson:** _(fill in)_

---

## Cross-cutting lessons

Record anything that applies across milestones here. Candidates:

- Where did the classifier learn shortcuts such as resolution or format?
- Which image types did it fail on most?
- What would break first if this were used on real uploads?

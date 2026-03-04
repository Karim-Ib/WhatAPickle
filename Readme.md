# WhatAPickle — CLIP-based Pickle Detector

A Discord-oriented image classifier that detects pickles in photos using OpenAI's CLIP model. Built to power a Discord bot that automatically deletes pickle-containing messages in a food channel (for trolling purposes).

The detector operates in two modes: **zero-shot** (no training data needed, uses text prompts) and **fine-tuned** (trained classifier head on CLIP embeddings, better accuracy). This README walks through the full workflow from setup to deployment.

---

## Project Structure

```
WhatAPickle/
├── pyproject.toml                 # Package config, dependencies, editable install
├── requirements.txt               # Dependency reference (pyproject.toml is source of truth)
├── README.md
├── src/
│   └── pickle_detector/
│       ├── __init__.py
│       ├── detector.py            # Core PickleDetector class (zero-shot + fine-tuned)
│       └── prompts.py             # Text prompts for zero-shot mode
├── scripts/
│   ├── scrape_images.py           # Download test images from Bing
│   ├── evaluate.py                # Run metrics, confusion matrix, threshold sweep
│   ├── test_single.py             # Test one image interactively
│   └── finetune.py                # Train classifier head on CLIP embeddings
├── data/
│   ├── pickle/                    # Labeled images WITH pickles
│   ├── non_pickle/                # Labeled images WITHOUT pickles
│   └── discord_validation/        # Optional: real Discord images for validation
│       ├── pickle/
│       └── non_pickle/
└── output/
    ├── pickle_head_logreg.pkl     # Trained classifier (after fine-tuning)
    ├── pickle_head_logreg.json    # Training metadata
    ├── embeddings_*.npz           # Cached CLIP embeddings
    └── score_distribution.png     # Eval visualization
```

---

## How It Works

### The Core Idea

CLIP (Contrastive Language-Image Pre-training) encodes both images and text into a shared embedding space. Two detection modes exploit this differently:

**Zero-shot mode** compares an image's embedding against text prompt embeddings like "a photo of pickles" and "a photo of food". The image gets classified based on which group of prompts it's most similar to. This requires no training data but is limited by what you can express in text — CLIP can't easily distinguish "burger with pickles" from "burger without pickles" through text alone.

**Fine-tuned mode** ignores text entirely. Instead, it extracts the image embedding from CLIP's vision encoder and feeds it through a trained logistic regression classifier. This classifier learned the decision boundary from your labeled images, so it can pick up on visual patterns (small green slices on a plate) that text prompts can't capture.

### Why CLIP Over a Standard Image Classifier?

Pre-trained classifiers (ResNet, EfficientNet on ImageNet) have ~1000 classes, and "pickle" isn't one of them. You'd need to fine-tune the entire network, which requires thousands of images and GPU training time. CLIP's embedding space already encodes rich visual concepts from internet-scale data — we just need a thin classifier on top, trained on ~100-300 images in seconds on CPU.

---

## Setup

### Requirements

- Python 3.9+
- ~2GB disk space (for CLIP model weights, cached on first run)
- No GPU required (CPU works fine, just slower)

### Installation

```bash
cd WhatAPickle
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -e .
pip install bing-image-downloader    # for scraping test images
```

The `pip install -e .` command installs the `pickle_detector` package in editable mode — you can modify the source code and changes take effect immediately without reinstalling.

### Thermal Warning

CLIP inference can spike CPU usage. If your machine runs hot or shuts down under load, add thread limiting. The evaluation script already does this, but if you write your own code:

```python
import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
import torch
torch.set_num_threads(2)
```

---

## Workflow

### Step 1: Build a Test Dataset

```bash
python scripts/scrape_images.py
```

This downloads ~80 pickle and ~120 non-pickle images from Bing into `data/pickle/` and `data/non_pickle/`.

**Then manually review both folders.** This step is critical — scraped images frequently have wrong labels. Common issues:

- `non_pickle/`: Burgers that actually have pickles in them, cucumber images that are actually pickled
- `pickle/`: Images where "pickle" appeared in the search query but the image is something else entirely

Delete or move mislabeled images. Aim for 50+ clean images per class minimum.

You can also add your own images (e.g., from your Discord food channel) — just drop them into the appropriate folder. The filename doesn't matter, only which folder it's in.

### Step 2: Sanity Check

Test individual images to build intuition for how the model scores things:

```bash
# Zero-shot mode (no trained head needed)
python scripts/test_single.py data/pickle/pickle_0001.jpg

# Shows: verdict, pickle score, and top matching text prompts
```

### Step 3: Evaluate Zero-Shot Baseline

```bash
python scripts/evaluate.py --sweep
```

This runs the detector against all labeled images and produces:

- Accuracy, precision, recall, F1
- Confusion matrix
- Failure analysis (which images it got wrong and why)
- Threshold sweep table (find the optimal decision boundary)
- Score distribution plot at `output/score_distribution.png`

The **score distribution plot** is the most informative output. Two well-separated histograms = the model works well. Overlapping histograms = the model struggles to tell the classes apart.

The **threshold sweep** shows precision/recall at different thresholds:

- Higher threshold (0.6-0.7) = fewer false positives (won't accidentally delete non-pickle images)
- Lower threshold (0.4-0.5) = higher recall (catches more pickles but also more false alarms)

### Step 4: Tune Prompts (Zero-Shot)

If the eval shows specific failure patterns, edit `src/pickle_detector/prompts.py`:

- **False negatives** (missed pickles): Add prompts describing what the missed images look like. Check the failure analysis to see which non-pickle prompt "won" — that tells you what CLIP confused the image with.
- **False positives** (non-pickles flagged): Add more non-pickle anchors for whatever is getting false-flagged. Cucumbers, zucchini, and green vegetables are common culprits.

Re-run `evaluate.py` after each change to measure impact. This is an iterative process.

### Step 5: Fine-Tune a Classifier Head

When prompt tuning hits its ceiling (typically around 90% F1), train a classifier:

```bash
python scripts/finetune.py
```

This extracts CLIP image embeddings for all labeled data (cached to disk), runs 5-fold cross-validation, and saves the trained model to `output/pickle_head_logreg.pkl`.

The cross-validation results are the honest performance estimate — the model is evaluated on images it never saw during training in each fold.

You can also try an MLP head:

```bash
python scripts/finetune.py --head mlp
```

In practice, logistic regression often outperforms MLP on small datasets (<500 images) because it's less prone to overfitting.

### Step 6: Evaluate the Fine-Tuned Model

```bash
python scripts/evaluate.py --head output/pickle_head_logreg.pkl
```

**Important**: Running this on the training data will show perfect or near-perfect scores (the model has seen these images). The cross-validation results from Step 5 are the real performance metric.

For a truly independent test, use images the model has never seen:

```bash
# Create a validation folder with new images
mkdir data\discord_validation\pickle
mkdir data\discord_validation\non_pickle
# Drop new images into the appropriate folders, then:
python scripts/evaluate.py --data-dir data/discord_validation --head output/pickle_head_logreg.pkl
```

### Step 7: Improve with More Data

If the fine-tuned model struggles with specific cases (e.g., pickles on a plate of food), the fix is almost always **more representative training data**:

1. Collect images that match your real use case (Discord food photos)
2. Label them into `data/pickle/` and `data/non_pickle/`
3. Delete the embedding cache: `del output\embeddings_ViT-B-32_laion2b_s34b_b79k.npz`
4. Retrain: `python scripts/finetune.py`
5. Re-evaluate against your validation set

The model learns its decision boundary from whatever you give it. Training on isolated pickle jars won't teach it to spot pickles hiding on a dinner plate.

---

## Using the Detector in Code

```python
from pickle_detector.detector import PickleDetector

# Zero-shot (no training data needed)
detector = PickleDetector(threshold=0.70)

# Fine-tuned (uses trained classifier head)
detector = PickleDetector(head_path="output/pickle_head_logreg.pkl")

# From file
result = detector.detect_file("path/to/image.jpg")
print(result.is_pickle)       # True/False
print(result.pickle_score)    # 0.0 - 1.0
print(result.mode)            # "zero_shot" or "finetuned"

# From PIL image
from PIL import Image
img = Image.open("photo.png")
result = detector.detect(img)

# From raw bytes (useful for Discord bot)
result = detector.detect_bytes(image_bytes)
```

---

## Configuration Reference

### CLIP Models

Specified via `--model` and `--pretrained` flags on all scripts.

| Model | Pretrained | Speed | Accuracy | Embedding Dim |
|-------|-----------|-------|----------|---------------|
| ViT-B-32 | laion2b_s34b_b79k | Fast | Good | 512 |
| ViT-L-14 | laion2b_s32b_b82k | ~4x slower | Better | 768 |

Start with ViT-B-32 for iteration. If you need better accuracy and can tolerate slower inference, try ViT-L-14. Note: if you switch models after fine-tuning, you need to retrain the head (embedding dimensions differ).

### Detection Modes

| Mode | Pros | Cons | When to Use |
|------|------|------|-------------|
| Zero-shot | No training data needed, easy to iterate on prompts | Limited by text-image matching, ~90% F1 ceiling | Quick start, prototyping |
| Fine-tuned (logreg) | Better accuracy, handles nuanced cases | Needs labeled data, retraining on new data | Production use |
| Fine-tuned (MLP) | Can learn non-linear boundaries | Overfits on small datasets | Only if you have 500+ images |

---

## Troubleshooting

**PC shuts down during evaluation**: CPU overheating under torch load. The eval and finetune scripts already limit threads to 2. If it still happens, run in Google Colab instead (free GPU, no thermal risk).

**`scrape_images.py` returns 0 URLs**: The Bing image downloader can be flaky. Try running again, or manually collect images from Google Images into `data/pickle/` and `data/non_pickle/`.

**Import errors / red underlines in IDE**: Make sure you ran `pip install -e .` from the project root. This installs the package in editable mode so your IDE can resolve `from pickle_detector import ...`.

**`TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`**: You're on Python 3.9, which doesn't support `X | Y` type hint syntax. This should already be fixed in all scripts — if you see it, replace `str | None` with `Optional[str]` and add `from typing import Optional`.

**Fine-tuned model shows 100% accuracy**: You're evaluating on the training data. The real performance is the cross-validation output from `finetune.py`. Use a separate validation folder with images the model hasn't seen for an independent test.

**Model struggles with food plates**: The training data probably has mostly isolated pickle images. Add more food-plate images (with and without pickles) to the training folders and retrain.
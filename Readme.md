# WhatAPickle — CLIP-based Pickle Detector

A Discord-oriented image classifier that detects pickles in photos using OpenAI's CLIP model, with an optional Gemini Vision API fallback for edge cases. Built to power a Discord bot that automatically deletes pickle-containing messages in a food channel (for trolling purposes).

The detector operates in three tiers:

1. **Zero-shot** — CLIP text-image similarity, no training needed
2. **Fine-tuned** — trained classifier head on CLIP embeddings, better accuracy
3. **Gemini fallback** — vision LLM API call for ambiguous cases the CLIP model can't confidently classify

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
│       ├── detector.py            # Core PickleDetector class (all 3 tiers)
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

### The Detection Pipeline

```
Image → CLIP embedding → Fine-tuned head → Score
                                              │
                              ┌────────────────┼────────────────┐
                              │                │                │
                         score > 0.7      0.3–0.7          score < 0.3
                          PICKLE        UNCERTAIN         NOT PICKLE
                                            │
                                     Gemini API call
                                            │
                                   ┌────────┴────────┐
                                PICKLE           NOT PICKLE
```

**Tier 1 — CLIP (local, free, fast):** The image is encoded by CLIP's vision encoder into a 512-dimensional embedding. This embedding is fed through a trained logistic regression classifier that outputs a pickle probability score between 0 and 1.

**Tier 2 — Gemini fallback (API, cheap, accurate):** If the CLIP score falls in an uncertain range (default 0.3–0.7), the image is sent to Google's Gemini Vision API for a second opinion. This handles edge cases like pickles partially hidden in a dish, pickle-shaped vegetables (zucchini, cucumber), and other ambiguous images.

This tiered approach keeps API costs minimal — only genuinely ambiguous images trigger an API call.

### Why CLIP Over a Standard Image Classifier?

Pre-trained classifiers (ResNet, EfficientNet on ImageNet) have ~1000 classes, and "pickle" isn't one of them. You'd need to fine-tune the entire network, which requires thousands of images and GPU training time. CLIP's embedding space already encodes rich visual concepts from internet-scale data — we just need a thin classifier on top, trained on ~100-300 images in seconds on CPU.

### Zero-Shot vs Fine-Tuned

**Zero-shot mode** compares an image's embedding against text prompt embeddings like "a photo of pickles" and "a photo of food". No training data needed, but limited by what you can express in text — CLIP can't easily distinguish "burger with pickles" from "burger without pickles" through text alone. Useful for prototyping. Tops out around 90% F1.

**Fine-tuned mode** ignores text entirely. It feeds the CLIP image embedding through a trained logistic regression classifier that learned the decision boundary from your labeled images. This picks up on visual patterns (small green slices on a plate) that text prompts can't capture. Our results: 91.3% accuracy, 92.5% precision, 86% recall on 5-fold cross-validation with ~300 images.

---

## Setup

### Requirements

- Python 3.9+
- ~2GB disk space (for CLIP model weights, cached on first run)
- No GPU required (CPU works fine, just slower)
- Google AI Studio API key (free, optional — only needed for Gemini fallback)

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
pip install google-genai             # for Gemini fallback (optional)
```

The `pip install -e .` command installs the `pickle_detector` package in editable mode — you can modify the source code and changes take effect immediately without reinstalling.

### Gemini API Key (Optional)

The Gemini fallback requires a free API key from Google AI Studio:

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Click "Create API Key" — no billing, no project setup needed
3. Set it as an environment variable (never hardcode it):

```bash
# Windows:
set GEMINI_API_KEY=your-key-here

# Linux/Mac:
export GEMINI_API_KEY=your-key-here
```

The detector uses `gemini-2.5-flash-lite` which has the most generous free tier limits: 15 requests/minute, 1000 requests/day.

### Thermal Warning

CLIP inference can spike CPU usage. If your machine runs hot or shuts down under load, add thread limiting. The evaluation and finetune scripts already do this, but if you write your own code:

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

Delete or move mislabeled images. Aim for 50+ clean images per class minimum. The filename doesn't matter, only which folder the image is in.

You can also add your own images (e.g., from your Discord food channel) — just drop them into the appropriate folder.

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

### Step 7: Test with Gemini Fallback

Once you have a fine-tuned head and a Gemini API key, test the full pipeline:

```bash
# Normal mode: Gemini only called if CLIP score is uncertain (0.3-0.7)
python scripts/test_single.py path/to/image.jpg --head output/pickle_head_logreg.pkl

# Force Gemini on every image (for testing the API connection)
python scripts/test_single.py path/to/image.jpg --head output/pickle_head_logreg.pkl --force-gemini

# Custom uncertainty range (wider = more API calls, higher accuracy)
python scripts/test_single.py path/to/image.jpg --head output/pickle_head_logreg.pkl --uncertain-low 0.2 --uncertain-high 0.8
```

The `GEMINI_API_KEY` environment variable is read automatically. You can also pass `--gemini-key` directly.

### Step 8: Improve with More Data

If the fine-tuned model struggles with specific cases (e.g., pickles on a plate of food, zucchini false positives), the fix is almost always **more representative training data**:

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

# Zero-shot only (no training data needed)
detector = PickleDetector(threshold=0.70)

# Fine-tuned head only (no API calls)
detector = PickleDetector(head_path="output/pickle_head_logreg.pkl")

# Fine-tuned + Gemini fallback (full pipeline)
import os
detector = PickleDetector(
    head_path="output/pickle_head_logreg.pkl",
    gemini_api_key=os.environ["GEMINI_API_KEY"],
    uncertain_range=(0.3, 0.7),
)

# From file
result = detector.detect_file("path/to/image.jpg")
print(result.is_pickle)          # True/False
print(result.pickle_score)       # 0.0 - 1.0
print(result.mode)               # "zero_shot", "finetuned", or "gemini_fallback"
print(result.gemini_reasoning)   # only populated when Gemini was called

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
| Gemini fallback | Highly accurate, handles edge cases | Requires API key, rate limited, adds latency | Combined with fine-tuned for edge cases |

### Gemini API Limits (Free Tier, as of early 2026)

| Model | RPM | RPD | Notes |
|-------|-----|-----|-------|
| gemini-2.5-flash-lite | 15 | 1,000 | **Used by default** — best free tier limits |
| gemini-2.5-flash | 10 | 250 | More capable, lower limits |
| gemini-2.5-pro | 5 | 100 | Most capable, very restricted |

The detector uses `gemini-2.5-flash-lite` by default. For a Discord bot that only sends uncertain images to the API, 1000 requests/day is more than enough.

### Uncertain Range Tuning

The `uncertain_range` parameter controls when Gemini is called:

| Range | Behavior | API Usage |
|-------|----------|-----------|
| (0.3, 0.7) | Default — moderate Gemini usage | Low |
| (0.2, 0.8) | More conservative — more API calls | Medium |
| (0.4, 0.6) | Aggressive — trusts CLIP more | Very low |
| (0.0, 1.0) | Every image goes to Gemini | High (testing only) |

For a troll bot, (0.3, 0.7) is a good starting point. Widen the range if you're seeing misclassifications on food plates, narrow it if API costs matter.

---

## Troubleshooting

**PC shuts down during evaluation**: CPU overheating under torch load. The eval and finetune scripts already limit threads to 2. If it still happens, run in Google Colab instead (free GPU, no thermal risk).

**`scrape_images.py` returns 0 URLs**: The Bing image downloader can be flaky. Try running again, or manually collect images from Google Images into `data/pickle/` and `data/non_pickle/`.

**Import errors / red underlines in IDE**: Make sure you ran `pip install -e .` from the project root. This installs the package in editable mode so your IDE can resolve `from pickle_detector import ...`.

**`TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'`**: You're on Python 3.9, which doesn't support `X | Y` type hint syntax. This should already be fixed in all scripts — if you see it, replace `str | None` with `Optional[str]` and add `from typing import Optional`.

**`ImportError: DLL load failed while importing _rust`**: The `cryptography` package version is incompatible with Python 3.9. Fix: `pip install cryptography==41.0.7`

**Fine-tuned model shows 100% accuracy**: You're evaluating on the training data. The real performance is the cross-validation output from `finetune.py`. Use a separate validation folder with images the model hasn't seen for an independent test.

**Model struggles with food plates**: The training data probably has mostly isolated pickle images. Add more food-plate images (with and without pickles) to the training folders and retrain.

**Gemini returns 429 (Too Many Requests)**: The free tier has strict rate limits. The detector has built-in retry logic with exponential backoff (3 attempts). If it still fails, wait a minute and try again. Newly created API keys sometimes need a moment to activate. If persistent, check your usage at [Google AI Studio](https://aistudio.google.com).

**Gemini 2.0 Flash not working**: This model was deprecated in February 2026 and retired March 3, 2026. The detector uses `gemini-2.5-flash-lite` by default, which has better free tier limits anyway.

---


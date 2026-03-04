"""
Fine-tune a classification head on frozen CLIP embeddings.

Why this works better than zero-shot:
- Zero-shot relies on text-image similarity, which means CLIP has to match
  "burger with pickles" as text against a burger image. The text encoder
  can't express the visual nuance of "there are small green slices visible".
- A trained classifier operates directly on the image embedding, which DOES
  contain that visual information — CLIP just can't surface it through text matching.
- We only train a tiny head (logistic regression or 1-hidden-layer MLP),
  so we need very little data and training takes seconds.

Workflow:
    1. Extract embeddings for all labeled images (cached to disk)
    2. Train/eval with stratified k-fold cross-validation
    3. Save the best model for use in the detector

Usage:
    python scripts/finetune.py                        # defaults (logistic regression)
    python scripts/finetune.py --head mlp             # small MLP instead
    python scripts/finetune.py --model ViT-L-14       # bigger CLIP backbone
    python scripts/finetune.py --no-cache             # re-extract embeddings
"""

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
torch.set_num_threads(2)

import open_clip
from PIL import Image
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def extract_embeddings(
        data_dir: Path,
        model_name: str,
        pretrained: str,
        device: str,
        cache_path: Optional[Path] = None,
) -> tuple:
    """
    Extract CLIP image embeddings for all labeled images.

    Returns (embeddings, labels, paths) where:
    - embeddings: (N, embed_dim) float32 array
    - labels: (N,) int array, 1=pickle, 0=non_pickle
    - paths: list of file paths for debugging
    """
    # Check cache first
    if cache_path and cache_path.exists():
        print(f"Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["embeddings"], data["labels"], data["paths"].tolist()

    print(f"Loading CLIP model {model_name} ({pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device
    )
    model.eval()

    embeddings = []
    labels = []
    paths = []

    for label_name, label_val in [("pickle", 1), ("non_pickle", 0)]:
        label_dir = data_dir / label_name
        if not label_dir.exists():
            continue

        img_files = sorted([
            p for p in label_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
        ])

        print(f"Extracting {label_name}: {len(img_files)} images")
        for img_path in tqdm(img_files, desc=f"  {label_name}"):
            try:
                image = Image.open(img_path).convert("RGB")
                img_tensor = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    embed = model.encode_image(img_tensor)
                    embed /= embed.norm(dim=-1, keepdim=True)

                embeddings.append(embed.cpu().numpy().squeeze())
                labels.append(label_val)
                paths.append(str(img_path))
            except Exception as e:
                print(f"    SKIP {img_path.name}: {e}")

    embeddings = np.array(embeddings, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    # Cache for fast re-runs
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, embeddings=embeddings, labels=labels, paths=np.array(paths))
        print(f"Cached embeddings to {cache_path}")

    return embeddings, labels, paths


def build_head(head_type: str, C: float = 1.0) -> Pipeline:
    """
    Build a classification pipeline.

    Pipeline always includes StandardScaler because CLIP embeddings are
    L2-normalized (unit sphere), but sklearn classifiers still benefit
    from per-feature scaling for stable optimization.
    """
    if head_type == "logreg":
        # Logistic regression — strong baseline, fast, interpretable.
        # L2 penalty with C=1.0 is a reasonable default.
        # max_iter=1000 because high-dimensional embeddings can need more steps.
        clf = LogisticRegression(
            C=C, max_iter=1000, solver="lbfgs", random_state=42
        )
    elif head_type == "mlp":
        # Small MLP — one hidden layer of 128 units.
        # This can capture non-linear decision boundaries that logreg can't,
        # which matters if "pickle on a burger" lives in a non-linearly
        # separable region of embedding space.
        clf = MLPClassifier(
            hidden_layer_sizes=(128,),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown head type: {head_type}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


def train_and_evaluate(
        embeddings: np.ndarray,
        labels: np.ndarray,
        head_type: str,
        n_folds: int = 5,
):
    """
    Stratified k-fold cross-validation.

    Why k-fold instead of a simple train/test split?
    With only ~300 images, a single 80/20 split is high-variance —
    you might get lucky or unlucky with which images land in test.
    5-fold gives a much more reliable estimate of real performance.
    """
    pipeline = build_head(head_type)

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    print(f"\nRunning {n_folds}-fold cross-validation with {head_type}...")
    results = cross_validate(
        pipeline, embeddings, labels,
        cv=cv, scoring=scoring, return_train_score=False,
    )

    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION RESULTS ({head_type}, {n_folds}-fold)")
    print(f"{'='*60}")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        scores = results[f"test_{metric}"]
        print(f"  {metric:>10}: {scores.mean():.3f} ± {scores.std():.3f}  {scores.round(3)}")

    return results


def train_final_model(
        embeddings: np.ndarray,
        labels: np.ndarray,
        head_type: str,
        save_path: Path,
):
    """Train on ALL data and save for deployment."""
    pipeline = build_head(head_type)
    pipeline.fit(embeddings, labels)

    # Quick sanity check — training accuracy
    train_preds = pipeline.predict(embeddings)
    train_acc = (train_preds == labels).mean()
    print(f"\nFinal model training accuracy: {train_acc:.3f}")

    # Save model
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Model saved to {save_path}")

    # Also save metadata for the detector to know which CLIP model to use
    meta = {
        "head_type": head_type,
        "n_samples": len(labels),
        "n_pickle": int(labels.sum()),
        "n_non_pickle": int((1 - labels).sum()),
        "train_accuracy": float(train_acc),
    }
    meta_path = save_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--head", type=str, default="logreg", choices=["logreg", "mlp"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Embedding cache path includes model name so different models don't collide
    cache_name = f"embeddings_{args.model}_{args.pretrained}.npz".replace("/", "_")
    cache_path = None if args.no_cache else (output_dir / cache_name)

    embeddings, labels, paths = extract_embeddings(
        data_dir, args.model, args.pretrained, device, cache_path
    )
    print(f"\nDataset: {len(labels)} images "
          f"({labels.sum()} pickle, {(1-labels).sum()} non-pickle), "
          f"embedding dim: {embeddings.shape[1]}")

    # Cross-validation
    train_and_evaluate(embeddings, labels, args.head, args.folds)

    # Train final model on all data
    model_path = output_dir / f"pickle_head_{args.head}.pkl"
    train_final_model(embeddings, labels, args.head, model_path)

    print(f"\nDone! To use in the detector:")
    print(f"  from pickle_detector.detector import PickleDetector")
    print(f"  detector = PickleDetector(head_path='{model_path}')")


if __name__ == "__main__":
    main()
"""
Evaluate the pickle detector against labeled test data.

Usage:
    # Zero-shot
    python scripts/evaluate.py --sweep

    # Fine-tuned head
    python scripts/evaluate.py --head output/pickle_head_logreg.pkl
"""

import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import argparse
import torch
torch.set_num_threads(2)

from pathlib import Path
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)
from tqdm import tqdm

from pickle_detector.detector import PickleDetector


@dataclass
class ImageResult:
    path: str
    true_label: str
    pickle_score: float
    predicted_pickle: bool
    correct: bool
    top_prompts: list


def load_dataset(data_dir: Path):
    samples = []
    for label in ["pickle", "non_pickle"]:
        label_dir = data_dir / label
        if not label_dir.exists():
            print(f"WARNING: {label_dir} not found")
            continue
        for p in sorted(label_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                samples.append((p, label))
    return samples


def evaluate(detector, samples):
    results = []
    for path, label in tqdm(samples, desc="Evaluating"):
        try:
            det = detector.detect_file(path)
        except Exception as e:
            print(f"  SKIP {path.name}: {e}")
            continue

        top3 = []
        if det.per_prompt_scores:
            sorted_prompts = sorted(det.per_prompt_scores.items(), key=lambda x: -x[1])
            top3 = [(p, f"{s:.4f}") for p, s in sorted_prompts[:3]]

        results.append(ImageResult(
            path=str(path),
            true_label=label,
            pickle_score=det.pickle_score,
            predicted_pickle=det.is_pickle,
            correct=(det.is_pickle == (label == "pickle")),
            top_prompts=top3,
        ))
    return results


def print_metrics(results, threshold):
    y_true = [1 if r.true_label == "pickle" else 0 for r in results]
    y_pred = [1 if r.pickle_score >= threshold else 0 for r in results]

    print(f"\n{'='*60}")
    print(f"RESULTS (threshold={threshold:.2f}, n={len(results)})")
    print(f"{'='*60}")
    print(f"  Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
    print(f"  Precision: {precision_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  Recall:    {recall_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"  F1:        {f1_score(y_true, y_pred, zero_division=0):.3f}")

    cm = confusion_matrix(y_true, y_pred)
    print(f"\n  Confusion Matrix:")
    print(f"                  Pred Non-Pickle  Pred Pickle")
    print(f"  True Non-Pickle    {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"  True Pickle        {cm[1][0]:>5}          {cm[1][1]:>5}")


def print_failures(results):
    failures = [r for r in results if not r.correct]
    if not failures:
        print("\nNo failures!")
        return

    print(f"\n{'='*60}")
    print(f"FAILURE ANALYSIS ({len(failures)} errors)")
    print(f"{'='*60}")

    fps = [r for r in failures if r.true_label == "non_pickle"]
    if fps:
        print(f"\n  FALSE POSITIVES ({len(fps)}):")
        for r in sorted(fps, key=lambda x: -x.pickle_score)[:10]:
            print(f"    score={r.pickle_score:.3f}  {Path(r.path).name}")
            if r.top_prompts:
                print(f"      top: {r.top_prompts[:2]}")

    fns = [r for r in failures if r.true_label == "pickle"]
    if fns:
        print(f"\n  FALSE NEGATIVES ({len(fns)}):")
        for r in sorted(fns, key=lambda x: x.pickle_score)[:10]:
            print(f"    score={r.pickle_score:.3f}  {Path(r.path).name}")
            if r.top_prompts:
                print(f"      top: {r.top_prompts[:2]}")


def plot_distributions(results, output_dir):
    pickle_scores = [r.pickle_score for r in results if r.true_label == "pickle"]
    non_pickle_scores = [r.pickle_score for r in results if r.true_label == "non_pickle"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(pickle_scores, bins=20, alpha=0.7, label="Pickle", color="green")
    axes[0].hist(non_pickle_scores, bins=20, alpha=0.7, label="Non-pickle", color="gray")
    axes[0].axvline(x=0.5, color="red", linestyle="--", label="Threshold")
    axes[0].set_xlabel("Pickle Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Score Distribution by Class")
    axes[0].legend()

    all_sorted = sorted(results, key=lambda r: r.pickle_score)
    colors = ["green" if r.true_label == "pickle" else "gray" for r in all_sorted]
    axes[1].bar(range(len(all_sorted)), [r.pickle_score for r in all_sorted], color=colors, width=1.0)
    axes[1].axhline(y=0.5, color="red", linestyle="--")
    axes[1].set_xlabel("Images (sorted by score)")
    axes[1].set_ylabel("Pickle Score")
    axes[1].set_title("Score Separation")

    plt.tight_layout()
    save_path = output_dir / "score_distribution.png"
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")


def threshold_sweep(results):
    y_true = np.array([1 if r.true_label == "pickle" else 0 for r in results])
    scores = np.array([r.pickle_score for r in results])

    print(f"\n{'='*60}")
    print("THRESHOLD SWEEP")
    print(f"{'='*60}")
    print(f"  {'Thresh':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")

    best_f1, best_t = 0, 0.5
    for t in np.arange(0.30, 0.80, 0.02):
        y_pred = (scores >= t).astype(int)
        f = f1_score(y_true, y_pred, zero_division=0)
        print(f"  {t:>8.2f} {accuracy_score(y_true, y_pred):>8.3f} "
              f"{precision_score(y_true, y_pred, zero_division=0):>8.3f} "
              f"{recall_score(y_true, y_pred, zero_division=0):>8.3f} {f:>8.3f}")
        if f > best_f1:
            best_f1, best_t = f, t

    print(f"\n  Best threshold: {best_t:.2f} (F1={best_f1:.3f})")
    return best_t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--head", type=str, default=None, help="Path to fine-tuned head .pkl")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    samples = load_dataset(data_dir)
    n_pickle = sum(1 for _, l in samples if l == "pickle")
    print(f"Loaded {len(samples)} images ({n_pickle} pickle, {len(samples) - n_pickle} non-pickle)")

    if not samples:
        print("No images found! Run: python scripts/scrape_images.py")
        return

    # Use threshold=0.5 for fine-tuned mode (logreg's natural decision boundary)
    threshold = 0.5 if args.head else args.threshold

    print(f"\nLoading model {args.model} ({args.pretrained})...")
    detector = PickleDetector(
        model_name=args.model,
        pretrained=args.pretrained,
        threshold=threshold,
        head_path=args.head,
    )
    print(f"Mode: {detector.mode}")

    results = evaluate(detector, samples)
    print_metrics(results, threshold)
    print_failures(results)

    if args.sweep:
        threshold_sweep(results)

    if not args.no_plot:
        plot_distributions(results, output_dir)


if __name__ == "__main__":
    main()
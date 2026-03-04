"""
Test a single image.

Usage:
    # Zero-shot
    python scripts/test_single.py path/to/image.jpg

    # Fine-tuned head
    python scripts/test_single.py path/to/image.jpg --head output/pickle_head_logreg.pkl
"""

import argparse
import sys
from pathlib import Path

from pickle_detector.detector import PickleDetector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--head", type=str, default=None, help="Path to fine-tuned head .pkl")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    path = Path(args.image)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Loading model {args.model}...")
    detector = PickleDetector(
        model_name=args.model,
        pretrained=args.pretrained,
        threshold=args.threshold,
        head_path=args.head,
    )

    print(f"Analyzing: {path.name} (mode: {detector.mode})")
    result = detector.detect_file(path)

    verdict = "PICKLE DETECTED" if result.is_pickle else "No pickle"
    print(f"\n  {verdict}")
    print(f"  Pickle score:     {result.pickle_score:.4f}")
    print(f"  Non-pickle score: {result.non_pickle_score:.4f}")
    print(f"  Mode:             {result.mode}")

    if result.per_prompt_scores:
        sorted_prompts = sorted(result.per_prompt_scores.items(), key=lambda x: -x[1])
        print(f"\n  Top {args.top_n} matching prompts:")
        for prompt, score in sorted_prompts[:args.top_n]:
            bar = "█" * int(score * 200)
            print(f"    {score:.4f}  {bar}  {prompt}")


if __name__ == "__main__":
    main()
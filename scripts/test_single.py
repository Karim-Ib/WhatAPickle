"""
Test a single image.

Usage:
    # CLIP only (zero-shot)
    python scripts/test_single.py path/to/image.jpg

    # CLIP with fine-tuned head
    python scripts/test_single.py path/to/image.jpg --head output/pickle_head_logreg.pkl

    # CLIP + Gemini fallback for edge cases
    python scripts/test_single.py path/to/image.jpg --head output/pickle_head_logreg.pkl --gemini-key YOUR_KEY

    # Custom uncertainty range
    python scripts/test_single.py path/to/image.jpg --head output/pickle_head_logreg.pkl --gemini-key YOUR_KEY --uncertain-low 0.2 --uncertain-high 0.8

    # Force Gemini regardless of CLIP score (for testing the API)
    python scripts/test_single.py path/to/image.jpg --gemini-key YOUR_KEY --force-gemini
"""

import os
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
    parser.add_argument("--head", type=str, default=None)
    parser.add_argument("--gemini-key", type=str, default=None,
                        help="Gemini API key. Also reads GEMINI_API_KEY env var.")
    parser.add_argument("--uncertain-low", type=float, default=0.3)
    parser.add_argument("--uncertain-high", type=float, default=0.7)
    parser.add_argument("--force-gemini", action="store_true",
                        help="Force Gemini call regardless of CLIP score (for testing)")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    path = Path(args.image)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    api_key = args.gemini_key or os.environ.get("GEMINI_API_KEY")

    # If --force-gemini, set range to (0.0, 1.0) so every score triggers fallback
    uncertain_range = (0.0, 1.0) if args.force_gemini else (args.uncertain_low, args.uncertain_high)

    print(f"Loading model {args.model}...")
    detector = PickleDetector(
        model_name=args.model,
        pretrained=args.pretrained,
        threshold=args.threshold,
        head_path=args.head,
        gemini_api_key=api_key,
        uncertain_range=uncertain_range,
    )

    print(f"Analyzing: {path.name}")
    result = detector.detect_file(path)

    verdict = "PICKLE DETECTED" if result.is_pickle else "No pickle"
    print(f"\n  {verdict}")
    print(f"  Pickle score:     {result.pickle_score:.4f}")
    print(f"  Non-pickle score: {result.non_pickle_score:.4f}")
    print(f"  Mode:             {result.mode}")

    if result.gemini_reasoning:
        print(f"  Gemini reasoning: {result.gemini_reasoning}")

    if result.per_prompt_scores:
        sorted_prompts = sorted(result.per_prompt_scores.items(), key=lambda x: -x[1])
        print(f"\n  Top {args.top_n} matching prompts:")
        for prompt, score in sorted_prompts[:args.top_n]:
            bar = "█" * int(score * 200)
            print(f"    {score:.4f}  {bar}  {prompt}")


if __name__ == "__main__":
    main()
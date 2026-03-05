"""
CLIP-based pickle detector — supports both zero-shot and fine-tuned modes.

Two detection modes:
1. Zero-shot (default): Compare image against text prompts via CLIP similarity.
   Good baseline, no training needed, but limited by text-image matching.
2. Fine-tuned head: Use a trained classifier on CLIP image embeddings.
   Better accuracy, especially for nuanced cases (pickle on a burger).

Usage:
    # Zero-shot
    detector = PickleDetector()

    # Fine-tuned
    detector = PickleDetector(head_path="output/pickle_head_logreg.pkl")
"""

import pickle as pkl
import open_clip
import torch
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from .prompts import PICKLE_PROMPTS, NON_PICKLE_PROMPTS


@dataclass
class DetectionResult:
    pickle_score: float
    non_pickle_score: float
    is_pickle: bool
    per_prompt_scores: dict    # only populated in zero-shot mode
    threshold: float
    mode: str                  # "zero_shot", "finetuned", or "zero_shot_and_finetuned"
    zero_shot_score: Optional[float] = None   # when mode is zero_shot_and_finetuned
    finetuned_score: Optional[float] = None   # when mode is zero_shot_and_finetuned


class PickleDetector:

    def __init__(
            self,
            model_name: str = "ViT-B-32",
            pretrained: str = "laion2b_s34b_b79k",
            device: Optional[str] = None,
            threshold: float = 0.55,
            head_path: Optional[str] = None,
            pickle_prompts: Optional[List[str]] = None,
            non_pickle_prompts: Optional[List[str]] = None,
    ):
        """
        head_path: path to a trained sklearn pipeline (.pkl file from finetune.py).
            If provided, uses the fine-tuned head instead of zero-shot.
            The threshold parameter is ignored in fine-tuned mode since
            the classifier has its own learned decision boundary.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.model.eval()

        # Always set up zero-shot (text prompts + embeddings)
        self.pickle_prompts = pickle_prompts or PICKLE_PROMPTS
        self.non_pickle_prompts = non_pickle_prompts or NON_PICKLE_PROMPTS
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self._pickle_embeds = self._encode_texts(self.pickle_prompts)
        self._non_pickle_embeds = self._encode_texts(self.non_pickle_prompts)
        self._all_embeds = torch.cat([self._pickle_embeds, self._non_pickle_embeds])
        self._n_pickle = len(self.pickle_prompts)

        # Load fine-tuned head if provided (then we run both zero-shot and fine-tuned)
        self.head = None
        if head_path:
            with open(head_path, "rb") as f:
                self.head = pkl.load(f)
            self.mode = "zero_shot_and_finetuned"
            print(f"Loaded fine-tuned head from {head_path} (using zero-shot + fine-tuned on every image)")
        else:
            self.mode = "zero_shot"

    def _encode_texts(self, texts: List[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            embeds = self.model.encode_text(tokens)
            embeds /= embeds.norm(dim=-1, keepdim=True)
        return embeds

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embed = self.model.encode_image(img_tensor)
            embed /= embed.norm(dim=-1, keepdim=True)
        return embed

    def detect(self, image: Image.Image) -> DetectionResult:
        img_embed = self._encode_image(image)

        if self.head:
            # Run both zero-shot and fine-tuned, combine with max(pickle_score)
            zs = self._detect_zero_shot(img_embed)
            ft = self._detect_finetuned(img_embed)
            combined_pickle = max(zs.pickle_score, ft.pickle_score)
            combined_non = min(zs.non_pickle_score, ft.non_pickle_score)
            return DetectionResult(
                pickle_score=combined_pickle,
                non_pickle_score=combined_non,
                is_pickle=combined_pickle >= self.threshold,
                per_prompt_scores=zs.per_prompt_scores,
                threshold=self.threshold,
                mode="zero_shot_and_finetuned",
                zero_shot_score=zs.pickle_score,
                finetuned_score=ft.pickle_score,
            )
        return self._detect_zero_shot(img_embed)

    def _detect_finetuned(self, img_embed: torch.Tensor) -> DetectionResult:
        """
        Fine-tuned mode: feed embedding through the sklearn pipeline.
        predict_proba gives us [p(non_pickle), p(pickle)].
        """
        embedding = img_embed.cpu().numpy()
        proba = self.head.predict_proba(embedding)[0]

        # sklearn orders classes as [0, 1] = [non_pickle, pickle]
        non_pickle_score = float(proba[0])
        pickle_score = float(proba[1])

        return DetectionResult(
            pickle_score=pickle_score,
            non_pickle_score=non_pickle_score,
            is_pickle=pickle_score >= 0.5,
            per_prompt_scores={},
            threshold=0.5,
            mode="finetuned",
        )

    def _detect_zero_shot(self, img_embed: torch.Tensor) -> DetectionResult:
        """Original zero-shot CLIP similarity scoring."""
        similarities = (100.0 * img_embed @ self._all_embeds.T).squeeze(0)
        probs = similarities.softmax(dim=-1)

        pickle_probs = probs[:self._n_pickle]
        non_pickle_probs = probs[self._n_pickle:]

        pickle_score = pickle_probs.sum().item()
        non_pickle_score = non_pickle_probs.sum().item()

        all_prompts = self.pickle_prompts + self.non_pickle_prompts
        per_prompt = {p: probs[i].item() for i, p in enumerate(all_prompts)}

        return DetectionResult(
            pickle_score=pickle_score,
            non_pickle_score=non_pickle_score,
            is_pickle=pickle_score >= self.threshold,
            per_prompt_scores=per_prompt,
            threshold=self.threshold,
            mode="zero_shot",
        )

    def detect_file(self, path: "str | Path") -> DetectionResult:
        image = Image.open(path).convert("RGB")
        return self.detect(image)

    def detect_bytes(self, data: bytes) -> DetectionResult:
        """Convenience for discord bot — accepts raw image bytes."""
        from io import BytesIO
        image = Image.open(BytesIO(data)).convert("RGB")
        return self.detect(image)
"""
CLIP-based pickle detector with optional LLM vision fallback.

Detection pipeline:
1. CLIP (zero-shot or fine-tuned head) scores the image — fast, free, local
2. If the score falls in an uncertain zone, escalate to a vision LLM API
   for a second opinion — slower, costs per call, but highly accurate

This keeps API costs minimal: only ambiguous images get sent to the LLM.

Usage:
    # CLIP only
    detector = PickleDetector(head_path="output/pickle_head_logreg.pkl")

    # CLIP + Gemini fallback for edge cases
    detector = PickleDetector(
        head_path="output/pickle_head_logreg.pkl",
        gemini_api_key="your-key-here",
        uncertain_range=(0.3, 0.7),
    )
"""

import os
import pickle as pkl
import io
import json
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
    per_prompt_scores: dict
    threshold: float
    mode: str  # "zero_shot", "finetuned", "zero_shot_and_finetuned", or "gemini_fallback"
    zero_shot_score: Optional[float] = None   # when mode is zero_shot_and_finetuned
    finetuned_score: Optional[float] = None   # when mode is zero_shot_and_finetuned
    gemini_reasoning: str = ""   # only populated when Gemini was called


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
            gemini_api_key: Optional[str] = None,
            uncertain_range: tuple = (0.3, 0.7),
    ):
        """
        gemini_api_key: Google AI Studio API key. If provided, images with
            scores inside uncertain_range get sent to Gemini for verification.

        uncertain_range: (low, high) tuple. Scores between these values
            are considered ambiguous and trigger the Gemini fallback.
            Default (0.3, 0.7) means:
            - score < 0.3 → confidently not pickle
            - score > 0.7 → confidently pickle
            - 0.3 to 0.7 → ask Gemini

        If gemini_api_key is not provided, GEMINI_API_KEY from the environment
        is used (e.g. from .env when loaded by the application).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.uncertain_low, self.uncertain_high = uncertain_range

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

        if gemini_api_key:
            print(f"Gemini fallback enabled for scores in ({self.uncertain_low}, {self.uncertain_high})")

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

    def _is_uncertain(self, score: float) -> bool:
        return self.uncertain_low <= score <= self.uncertain_high

    def _ask_gemini(self, image: Image.Image) -> DetectionResult:
        """
        Send image to Gemini for pickle classification.

        Uses a structured prompt that asks for:
        - A yes/no pickle verdict
        - A confidence score
        - Brief reasoning (useful for debugging)

        Returns a DetectionResult with mode="gemini_fallback".
        """
        from google import genai

        client = genai.Client(api_key=self.gemini_api_key)

        # Convert PIL image to bytes for the SDK
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()

        prompt = (
            "Does this image contain pickles (pickled cucumbers)? "
            "This includes pickle slices on burgers, sandwiches, pizza, "
            "or any dish, as well as whole pickles, pickle jars, pickle "
            "spears, gherkins, and pickle-themed memes or cartoons.\n\n"
            "Fresh raw cucumbers are NOT pickles.\n\n"
            "Respond with ONLY valid JSON, no markdown:\n"
            '{"contains_pickle": true/false, "confidence": 0.0-1.0, '
            '"reasoning": "brief explanation"}'
        )

        try:
            import time
            import random

            last_error = None
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash-lite",
                        contents=[
                            {
                                "parts": [
                                    {"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}},
                                    {"text": prompt},
                                ]
                            }
                        ],
                        config={
                            "temperature": 0.1,
                            "max_output_tokens": 150,
                        },
                    )
                    break  # success
                except Exception as e:
                    last_error = e
                    if "429" in str(e) and attempt < 2:
                        wait = (2 ** attempt) + random.uniform(0, 1)
                        print(f"  Rate limited, retrying in {wait:.1f}s (attempt {attempt + 1}/3)")
                        time.sleep(wait)
                    else:
                        raise

            text = response.text.strip().strip("`").strip()
            if text.startswith("json"):
                text = text[4:].strip()

            result = json.loads(text)

            is_pickle = result.get("contains_pickle", False)
            confidence = float(result.get("confidence", 0.5))
            reasoning = result.get("reasoning", "")

            pickle_score = confidence if is_pickle else (1.0 - confidence)

            return DetectionResult(
                pickle_score=pickle_score,
                non_pickle_score=1.0 - pickle_score,
                is_pickle=is_pickle,
                per_prompt_scores={},
                threshold=0.5,
                mode="gemini_fallback",
                gemini_reasoning=reasoning,
            )

        except Exception as e:
            # If Gemini fails, fall back to CLIP score
            print(f"Gemini API error: {e}. Falling back to CLIP score.")
            return None

    def detect(self, image: Image.Image) -> DetectionResult:
        """
        Main detection pipeline:
        1. Run CLIP (fast, local)
        2. If score is uncertain AND Gemini is configured → ask Gemini
        3. Otherwise return CLIP result
        """
        img_embed = self._encode_image(image)

        if self.head:
            # Run both zero-shot and fine-tuned, combine with max(pickle_score)
            zs = self._detect_zero_shot(img_embed)
            ft = self._detect_finetuned(img_embed)
            combined_pickle = max(zs.pickle_score, ft.pickle_score)
            combined_non = min(zs.non_pickle_score, ft.non_pickle_score)
            clip_result = DetectionResult(
                pickle_score=combined_pickle,
                non_pickle_score=combined_non,
                is_pickle=combined_pickle >= self.threshold,
                per_prompt_scores=zs.per_prompt_scores,
                threshold=self.threshold,
                mode="zero_shot_and_finetuned",
                zero_shot_score=zs.pickle_score,
                finetuned_score=ft.pickle_score,
                gemini_reasoning="",
            )
        else:
            clip_result = self._detect_zero_shot(img_embed)

        # Escalate to Gemini if score is in uncertain range
        if self.gemini_api_key and self._is_uncertain(clip_result.pickle_score):
            gemini_result = self._ask_gemini(image)
            if gemini_result is not None:
                return gemini_result

        return clip_result

    def _detect_finetuned(self, img_embed: torch.Tensor) -> DetectionResult:
        embedding = img_embed.cpu().numpy()
        proba = self.head.predict_proba(embedding)[0]

        non_pickle_score = float(proba[0])
        pickle_score = float(proba[1])

        return DetectionResult(
            pickle_score=pickle_score,
            non_pickle_score=non_pickle_score,
            is_pickle=pickle_score >= 0.5,
            per_prompt_scores={},
            threshold=0.5,
            mode="finetuned",
            gemini_reasoning="",
        )

    def _detect_zero_shot(self, img_embed: torch.Tensor) -> DetectionResult:
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
            gemini_reasoning="",
        )

    def detect_file(self, path: "str | Path") -> DetectionResult:
        image = Image.open(path).convert("RGB")
        return self.detect(image)

    def detect_bytes(self, data: bytes) -> DetectionResult:
        """Convenience for discord bot — accepts raw image bytes."""
        from io import BytesIO
        image = Image.open(BytesIO(data)).convert("RGB")
        return self.detect(image)
from __future__ import annotations

import torch
from transformers import pipeline
from typing import Dict


class TextModel:
    """Emotion classification across the text channel."""

    def __init__(self, lang: str = "en", device: torch.device | None = None) -> None:
        """
        Args:
            lang (str, optional): Speech language for text processing ['ru', 'en']. Defaults to 'en'.
            device (torch.device | None, optional): Device type on local machine (GPU recommended). Defaults to None.
        """
        super().__init__()
        if lang not in ["en", "ru"]:
            raise NotImplementedError("'lang' must be 'en' or 'ru'.")

        self._lang = lang
        self._device = torch.device("cpu")
        if device is not None:
            self._device = device
        self.tokenizer_kwargs = {"padding": True, "truncation": True}

        # Initialization of the ruBert Russian-language model for emotion classification.
        if self._lang == "ru":
            self.classifier = pipeline(
                task="text-classification",
                model="Aniemore/rubert-tiny2-russian-emotion-detection",
                tokenizer="Aniemore/rubert-tiny2-russian-emotion-detection",
                framework="pt",
                function_to_apply="softmax",
                return_all_scores=True,
                device=self._device
            )

        # Initialization of the DistilRoBERTa English-language model for emotion classification.
        if self._lang == "en":
            self.classifier = pipeline(
                task="text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                tokenizer="j-hartmann/emotion-english-distilroberta-base",
                framework="pt",
                function_to_apply="softmax",
                return_all_scores=True,
                device=self._device
            )

    @property
    def device(self) -> torch.device:
        """Check the device type.

        Returns:
            torch.device: Device type on local machine.
        """
        return self._device

    @torch.no_grad()
    def predict(self, text: str) -> Dict:
        """Predicting emotions from the text in English or Russian.

        Args:
            text (str): The text to classify.
        """
        assert isinstance(text, str), "Text must be string type."

        emotion_dict = {"anger": 0., "neutral": 0., "happiness": 0.}

        if self._lang == "ru":
            output = self.classifier(text, **self.tokenizer_kwargs)

            emotion_dict["anger"] = output[0][5]["score"]
            emotion_dict["neutral"] = output[0][0]["score"]
            emotion_dict["happiness"] = output[0][1]["score"]

        elif self._lang == "en":
            output = self.classifier(text, **self.tokenizer_kwargs)

            emotion_dict["anger"] = output[0][0]["score"]
            emotion_dict["neutral"] = output[0][4]["score"]
            emotion_dict["happiness"] = output[0][3]["score"]

        return emotion_dict

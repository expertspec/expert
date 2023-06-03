from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
from transformers import pipeline


class DistortionModel:
    """Congitive distortions classification across the text channel."""

    def __init__(
        self,
        lang: Optional[str] = "en",
        device: Optional[Union[torch.device, None]] = None,
    ) -> None:
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
        self.tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "max_length": 128,
        }

        # Initialization of the ruBert Tiny2 Russian-language model for distortions classification.
        if self._lang == "ru":
            self.classifier = pipeline(
                task="text-classification",
                model="amedvedev/rubert-tiny2-cognitive-bias",
                tokenizer="amedvedev/rubert-tiny2-cognitive-bias",
                framework="pt",
                top_k=8,
                function_to_apply="softmax",
                device=self._device,
            )

        # Initialization of the Bert Tiny English-language model for distortions classification.
        if self._lang == "en":
            self.classifier = pipeline(
                task="text-classification",
                model="amedvedev/bert-tiny-cognitive-bias",
                tokenizer="amedvedev/bert-tiny-cognitive-bias",
                framework="pt",
                top_k=8,
                function_to_apply="softmax",
                device=self._device,
            )

    @property
    def device(self) -> torch.device:
        """Check the device type.

        Returns:
            torch.device: Device type on local machine.
        """
        return self._device

    @torch.no_grad()
    def predict(self, text: str) -> List[Dict]:
        """Predicting distortions from the text in English or Russian.

        Args:
            text (str): The text to classify.
        """
        assert isinstance(text, str), "Text must be string type."

        output = self.classifier(text, **self.tokenizer_kwargs)[0]

        return output

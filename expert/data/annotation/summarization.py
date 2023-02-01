from __future__ import annotations

import torch
from transformers import pipeline
import gdown
import os

from expert.core.utils import get_torch_home


class Summarization:
    def __init__(self, device: torch.device | None = None, max_length: int = 25, summary_percent: int = 21):
        self._device = torch.device("cpu")
        if device is not None:
            self._device = device
        
        url = "https://drive.google.com/drive/folders/1Taqf-72HoZykGuW29HtEfMMVUX1S0esu"
        model_name = "bart_large"
        model_dir = os.path.join(get_torch_home(), "checkpoints")
        os.makedirs(model_dir, exist_ok=True)
        
        cached_dir = os.path.join(model_dir, model_name)
        if not os.path.exists(cached_dir):
            gdown.download_folder(url, output=cached_dir, quiet=False)
        
        self.summarizer_bart = pipeline("summarization", model=cached_dir, device=self._device)
        self.summary_percent = summary_percent
        self.max_length = 25
    
    @property
    def device(self) -> torch.device:
        """Check the device type."""
        return self._device
    
    def get_summary(self, text: str, ):
        """Get summary for short text, paragraph (512 words).
        
        Args:
            text (str): Text for which we generate summarization.
            summary_percent (int, optional): Maximum summary size from source text. Defaults to 21.
        
        Returns:
            str: Summary of text.
        """
        text = text.strip()
        
        text25 = int(len(text.split()) / 100 * self.summary_percent)
        max_length = self.max_length if text25 < self.max_length else text25
        
        return self.summarizer_bart(
            text, min_length=5, max_length=self.max_length, do_sample=False
        )[0]["summary_text"]
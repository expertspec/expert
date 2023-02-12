from __future__ import annotations

import torch
from transformers import pipeline
import gdown
import os

from expert.core.utils import get_model_folder


class Summarization:
    """Annotating input text to a given size.
    
    Implementation of "BART: Denoising Sequence-to-Sequenceи Pre-training
    for Natural Language Generation, Translation, and Comprehension",
    https://arxiv.org/pdf/1910.13461, for annotating English text to a given size.
    
    Args:
        device (torch.device | None, optional): Device type on local machine (GPU recommended). Defaults to None.
        max_length (int, optional): Maximum number of tokens in the generated text. Defaults to 25.
        summary_percent (int, optional): Maximum annotation percentage of original text size. Defaults to 25.
    """
    def __init__(
        self,
        device: torch.device | None = None,
        max_length: int = 25,
        summary_percent: int = 25
    ):
        self._device = torch.device("cpu")
        if device is not None:
            self._device = device
        
        # Download weights for selected model if missing.
        url = "https://drive.google.com/drive/folders/1Taqf-72HoZykGuW29HtEfMMVUX1S0esu"
        model_name = "bart_large"
        cached_dir = get_model_folder(model_name=model_name, url=url)
        
        self.summarizer_bart = pipeline("summarization", model=cached_dir, device=self._device)
        self.max_length = 25
        self.summary_percent = summary_percent
    
    @property
    def device(self) -> torch.device:
        """Check the device type.
        
        Returns:
            torch.device: Device type on local machine.
        """
        return self._device
    
    def get_summary(self, text: str):
        """Generate annotation for a short text (512 tokens max).
        
        Args:
            text (str): Original text for annotation generation.
        
        Returns:
            str: Annotation of text.
        """
        text = text.strip()
        text_percent = int(len(text.split()) / 100 * self.summary_percent)
        max_length = self.max_length if text_percent < self.max_length else text_percent
        generated_text = self.summarizer_bart(
            text,
            min_length=5,
            max_length=self.max_length,
            do_sample=False
        )[0]["summary_text"]
        
        return generated_text
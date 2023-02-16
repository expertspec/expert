from __future__ import annotations

import torch
from transformers import pipeline
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from text_preprocessing import remove_email, remove_url
from razdel import sentenize
from typing import List, Tuple
import gdown
import os

from expert.core.functional_tools import get_model_folder


class SummarizationEN:
    """Annotating input text to a given size in English.
    
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
            max_length=max_length,
            do_sample=False
        )[0]["summary_text"]
        
        return generated_text


class SummarizationRU:
    """Annotating input text to a given size in Russian.
    
    Implementation of "LexRank: Graph-based Lexical Centrality as Salience in
    Text Summarization", https://arxiv.org/abs/1109.2128, for annotating English text.
    """
    def _delete_nonfunctional_dots(self, text: str) -> str:
        return " ".join([sentence.text[:-1].replace(".", "") + sentence.text[-1] for sentence in list(sentenize(text)) if len(sentence.text) > 0])
    
    def _delete_empty_brackets(self, text: str) -> str:
        text = text.replace("()", "").replace("( )", "").replace("(  )", "")
        
        return text
    
    def _cleanup_punctuation(self, text: str, allowed_punctuation: List) -> str:
        clean_text = ""
        for ch in text:
            if ch.isalpha() or ch.isdigit() or (ch in allowed_punctuation + [" "]):
                clean_text += ch
        
        return clean_text
    
    def _cut_title(self, text: str, max_length: int, over_chared_postfix: str, cutted_end_punct: List = ["."]) -> str:
        words = text.split()
        ch_count = 0
        left_words = []
        over_chared = False
        for word in words:
            if ch_count + len(word) + len(over_chared_postfix) <= max_length:
                left_words.append(word)
                # Add 'space' to number of characters.
                ch_count += len(word) + 1
            else:
                over_chared = True
                break
        answer = " ".join(left_words) + (over_chared_postfix if over_chared else "")
        if (not over_chared) and (answer[-1] in cutted_end_punct):
            answer = answer[:-1]
        
        return answer
    
    def _bracket_trouble(self, text: str) -> str:
        if "(" in text and ")" not in text:
            text = text.replace("(", "")
        if ")" in text and "(" not in text:
            text = text.replace(")", "")
        
        return text
    
    def _to_lower_register(self, text: str, threshold: int = 2) -> str:
        if len(text.split()) > threshold and text.upper() == text:
            return text.capitalize()
        return text
    
    def _preprocess(self, text: str) -> str:
        text = remove_url(text)
        text = text.strip()
        text = self._delete_nonfunctional_dots(text)
        
        return text
    
    def _postprocess(
        self,
        text: str,
        context: Tuple,
        max_length: int,
        over_chared_postfix: str,
        allowed_punctuation: List
    ) -> str:
        text = text.strip()
        text = self._delete_empty_brackets(text)
        text = self._cleanup_punctuation(text, allowed_punctuation)
        text = self._bracket_trouble(text)
        text = self._cut_title(text, max_length, over_chared_postfix)
        text = self._to_lower_register(text)
        
        return text

    def get_summary(
        self,
        text: str,
        sentences_count: int = 2,
        max_length: int = 300,
        over_chared_postfix: str = "...",
        allowed_punctuation: List = [",", ".", "!", "?", ":", "—", "-", "#", "+",
                                     "(", ")", "–", "%", "&", "@", '"', "'",]
    ) -> str:
        """Get annotation for a text.
        
        Args:
            text (str): Original text.
            sentences_count (int, optional): Sentences count in output annotation. Defaults to 2.
            max_length (int, optional): Maximum number of symbols in output annotation. Defaults to 300.
            over_chared_postfix (str, optional): End of line character when truncated. Defaults to "...".
            allowed_punctuation (List, optional): Allowed punctuation.
        
        Returns:
            str: Annotation of text.
        """
        text = self._preprocess(text)
        
        my_parser = PlaintextParser.from_string(text, Tokenizer("russian"))
        lex_rank_summarizer = LexRankSummarizer()
        lexrank_summary = lex_rank_summarizer(
            my_parser.document, sentences_count=sentences_count
        )
        
        dirty_summary = ""
        for s in lexrank_summary:
            dirty_summary += s._text + " "
        dirty_summary = dirty_summary.rstrip()
        
        summary = self._postprocess(
            dirty_summary, my_parser.document.sentences,
            max_length, over_chared_postfix, allowed_punctuation
        )
        
        return summary
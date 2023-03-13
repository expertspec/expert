from __future__ import annotations

from abc import ABC, abstractmethod
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
from text_preprocessing import remove_url
from typing import List, Tuple

from expert.core.functional_tools import get_model_folder


class Summarization(ABC):
    """Abstract class for Annotating input text to a given size. 
    Contains implementation of preprocessing and postprocessing methods.
    """

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
        answer = " ".join(left_words) + \
            (over_chared_postfix if over_chared else "")
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
        text = self._bracket_trouble(text)
        text = self._cut_title(text, max_length, over_chared_postfix)
        text = self._to_lower_register(text)

        return text

    @abstractmethod
    def get_summary(self, text: str, sentences_count: int):
        pass


class SummarizationEN(Summarization):
    """Annotating input text to a given size in English.

    Implementation of "Text summarization using Latent Semantic Analysis",
      https://www.researchgate.net/publication/220195824_Text_summarization_using_Latent_Semantic_Analysis, for annotating English text.
    """

    def get_summary(
        self,
        text: str,
        sentences_count: int = 1,
        max_length: int = 300,
        over_chared_postfix: str = "...",
        allowed_punctuation: List = [",", ".", "!", "?", ":", "—", "-", "#", "+",
                                     "(", ")", "–", "%", "&", "@", '"', "'"]
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

        stemmer = Stemmer("english")
        lsa_summarizer = LsaSummarizer(stemmer)
        lsa_summarizer.stop_words = get_stop_words("english")

        my_parser = PlaintextParser.from_string(text, Tokenizer('english'))
        lsa_summary = lsa_summarizer(
            my_parser.document, sentences_count=sentences_count
        )

        dirty_summary = ""
        for s in lsa_summary:
            dirty_summary += s._text + " "

        dirty_summary = dirty_summary.rstrip()

        summary = self._postprocess(
            text=dirty_summary, context=my_parser.document.sentences,
            max_length=max_length, over_chared_postfix=over_chared_postfix, allowed_punctuation=allowed_punctuation
        )

        return summary


class SummarizationRU(Summarization):
    """Annotating input text to a given size in Russian.

    Implementation of "Text summarization using Latent Semantic Analysis",
      https://www.researchgate.net/publication/220195824_Text_summarization_using_Latent_Semantic_Analysis, for annotating Russian text.
    """

    def get_summary(
        self,
        text: str,
        sentences_count: int = 1,
        max_length: int = 300,
        over_chared_postfix: str = "...",
        allowed_punctuation: List = [",", ".", "!", "?", ":", "—", "-", "#", "+",
                                     "(", ")", "–", "%", "&", "@", '"', "'"]
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
        lsa_summarizer = LsaSummarizer()
        lsa_summary = lsa_summarizer(
            my_parser.document, sentences_count=sentences_count
        )

        dirty_summary = ""
        for s in lsa_summary:
            dirty_summary += s._text + " "
        dirty_summary = dirty_summary.rstrip()

        summary = self._postprocess(
            text=dirty_summary, context=my_parser.document.sentences,
            max_length=max_length, over_chared_postfix=over_chared_postfix, allowed_punctuation=allowed_punctuation
        )

        return summary

from __future__ import annotations

import re
from typing import List, Optional

import nltk
import torch
from detoxify import Detoxify
from flair.data import Sentence
from flair.models import SequenceTagger
from nltk import RegexpParser, pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tree import Tree


try:
    nltk.data.find("taggers/averaged_perceptron_tagger")
except LookupError:
    nltk.download("averaged_perceptron_tagger")


class DepreciationEN:
    def __init__(self) -> None:
        self.pstemmer = PorterStemmer()
        self.tagger = SequenceTagger.load("flair/pos-english-fast")

    def _word_depreciation(self, word: str, sent: str) -> Optional[str]:
        word_re = re.search(r"((let)|(ule)|(ette)|(ock))s?\b", word)
        word_re_ing = re.search(r"((king)|(ling))s?\b", word)
        if word_re:
            return word
        elif word_re_ing:
            if not self.is_verb(word, sent):
                return word
        return None

    def is_verb(self, word: str, sent: str) -> bool:
        verb_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

        word_st = self.pstemmer.stem(word)
        sent = self.pstemmer.stem(sent)

        sentence = Sentence(sent)
        self.tagger.predict(sentence)

        for token in sentence:
            if (word_st == token.text) or (word == token.text):
                print(token)
                if token.get_label("pos").value in verb_tags:
                    return True
                return False
        return True

    def is_depreciation(self, sent: str) -> List:
        depr_words = []
        tokens = word_tokenize(sent)
        for token in tokens:
            word_depr = self._word_depreciation(token, sent)
            if word_depr:
                depr_words.append(word_depr)

        return len(depr_words)


class ToxicEN:
    def __init__(self, device: torch.device | None = None) -> None:
        self._device = torch.device("cpu")
        if device is not None:
            self._device = device

        self.model = Detoxify("original", device=self._device)

    def is_toxic(self, sent: str) -> bool:
        res_tox = self.model.predict(sent)
        if res_tox["toxicity"] >= 0.5:
            return True
        else:
            return False


class ImperativeEN:
    def _get_chunks(self, tagged_sent: List) -> Tree:
        # Chunk the sentence into grammatical phrases based on its POS-tags.
        chunkgram = r"""VB-Phrase: {<DT><,>*<VB>}
                        VB-Phrase: {<RB><VB>}
                        VB-Phrase: {<NN><RB><VB>}
                        VB-Phrase: {<UH><,>*<VB>}
                        VB-Phrase: {<UH><,>*<VBP>}
                        VB-Phrase: {<PRP><VB>}
                        VB-Phrase: {<PRP><VBP>}
                        VB-Phrase: {<NN.?>+<,>*<VB>}
                        VB-Phrase: {<NN.?>+<,>*<VBP>}
                        VB-Phrase: {<NN.?>+<,>*<PRP><VB>}
                        VB-Phrase: {<NN.?>+<,>*<PRP><VBP>}
                        Q-Tag: {<,><MD><RB>*<PRP><.>*}"""
        chunkparser = RegexpParser(chunkgram)
        return chunkparser.parse(tagged_sent)

    def _check_imperative(self, tagged_sent: List) -> bool:
        # If it's a request.
        if "please" in self.sent:
            return False

        chunk = self._get_chunks(tagged_sent)
        # Сatches imperatives ending with a Question Tag and
        # starting with a verb in base form, e.g. "Stop it, will you?".
        if type(chunk[-1]) is Tree and chunk[-1].label() == "Q-Tag":
            if chunk[0][1] == "VB" or (
                type(chunk[0]) is Tree and chunk[0].label() == "VB-Phrase"
            ):
                return True
        return False

    def is_imperative(self, sent: str) -> bool:
        """Checks if a sentence is imperative.

        Args:
            sent (str): Target text sentence.

        Returns:
            bool: True if it is an imperative sentence otherwise False.
        """
        sent = sent.replace("n't", " not")
        self.sent = word_tokenize(sent)
        tokens_tag = pos_tag(self.sent, lang="eng")
        res_impr = self._check_imperative(tokens_tag)

        return res_impr

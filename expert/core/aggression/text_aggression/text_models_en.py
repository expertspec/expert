import re
from typing import Optional

import nltk
from nltk.tree import Tree
from nltk import RegexpParser, pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from flair.data import Sentence
from flair.models import SequenceTagger

import torch
from detoxify import Detoxify


nltk.download('averaged_perceptron_tagger')


class Depreciation:
    def __init__(self) -> None:
        self.pstemmer = PorterStemmer()
        self.tagger = SequenceTagger.load("flair/pos-english-fast")

    def _word_deprication(self, word: str, sent: str) -> Optional[str]:
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

        # predict pos tags
        self.tagger.predict(sentence)

        for token in sentence:
            if (word_st == token.text) or (word == token.text):
                print(token)
                if token.get_label("pos").value in verb_tags:
                    return True
                return False

        return True

    def is_deprication(self, sent: str) -> list():
        depr_words = []

        tokens = word_tokenize(sent)

        for token in tokens:
            word_depr = self._word_deprication(token, sent)
            if word_depr:
                depr_words.append(word_depr)

        return len(depr_words)


class Toxic:

    def __init__(self) -> None:
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.model = Detoxify('original', device=self.device)

    def is_toxic(self, sent: str) -> bool:

        res_tox = self.model.predict(sent)
        if res_tox['toxicity'] >= 0.5:
            return True
        else:
            return False

class Imperative():

    def _get_chunks(self, tagged_sent):
        # chunks the sentence into grammatical phrases based on its POS-tags
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

    def _check_imperative(self, tagged_sent):
        # если это просьба
        if 'please' in self.sent:
            return False
        # Questions can be imperatives too, let's check if this one is
        
        # check if sentence contains the word 'please'
        # pls = len([w for w in tagged_sent if w[0].lower() == "please"]) > 0
        # catches requests disguised as questions
        # e.g. "Open the doors, HAL, please?"
        # if pls and (tagged_sent[0][1] == "VB" or tagged_sent[0][1] == "MD"):
        #     return True

        chunk = self._get_chunks(tagged_sent)
        # catches imperatives ending with a Question tag
        # and starting with a verb in base form, e.g. "Stop it, will you?"
        if type(chunk[-1]) is Tree and chunk[-1].label() == "Q-Tag":
            if (chunk[0][1] == "VB" or
                    (type(chunk[0]) is Tree and chunk[0].label() == "VB-Phrase")):
                return True

        return False

    def is_imperative(self, sent: str) -> bool:
        """Проверяет находится ли предложение в повелительном наклонении

        Args:
            sent (str): Предложение

        Returns:
            bool: True если это повелительное предложение иначе False
        """
        sent = sent.replace("n't", ' not')
        self.sent = word_tokenize(sent)
        tokens_tag = pos_tag(self.sent, lang='eng')
        res_impr = self._check_imperative(tokens_tag)
        return res_impr
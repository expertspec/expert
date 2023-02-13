import re
import pymorphy2

import torch
from transformers import BertTokenizer, BertForSequenceClassification


class Porter:
    PERFECTIVEGROUND = re.compile(
        u"((ив|ивши|ившись|ыв|ывши|ывшись)|((?<=[ая])(в|вши|вшись)))$")
    REFLEXIVE = re.compile(u"(с[яь])$")
    ADJECTIVE = re.compile(
        u"(ее|ие|ые|ое|ими|ыми|ей|ий|ый|ой|ем|им|ым|ом|его|ого|ему|ому|их|ых|ую|юю|ая|яя|ою|ею)$")
    PARTICIPLE = re.compile(u"((ивш|ывш|ующ)|((?<=[ая])(ем|нн|вш|ющ|щ)))$")
    VERB = re.compile(
        u"((ила|ыла|ена|ейте|уйте|ите|или|ыли|ей|уй|ил|ыл|им|ым|ен|ило|ыло|ено|ят|ует|уют|ит|ыт|ены|ить|ыть|ишь|ую|ю)|((?<=[ая])(ла|на|ете|йте|ли|й|л|ем|н|ло|но|ет|ют|ны|ть|ешь|нно)))$")
    NOUN = re.compile(
        u"(а|ев|ов|ие|ье|е|иями|ями|ами|еи|ии|и|ией|ей|ой|ий|й|иям|ям|ием|ем|ам|ом|о|у|ах|иях|ях|ы|ь|ию|ью|ю|ия|ья|я)$")
    RVRE = re.compile(u"^(.*?[аеиоуыэюя])(.*)$")
    DERIVATIONAL = re.compile(u".*[^аеиоуыэюя]+[аеиоуыэюя].*ость?$")
    DER = re.compile(u"ость?$")
    SUPERLATIVE = re.compile(u"(ейше|ейш)$")
    I = re.compile(u"и$")
    P = re.compile(u"ь$")
    NN = re.compile(u"нн$")

    def stem(word):
        word = word.lower()
        word = word.replace(u'ё', u'е')
        m = re.match(Porter.RVRE, word)
        if m and m.groups():
            pre = m.group(1)
            rv = m.group(2)
            temp = Porter.PERFECTIVEGROUND.sub('', rv, 1)
            if temp == rv:
                rv = Porter.REFLEXIVE.sub('', rv, 1)
                temp = Porter.ADJECTIVE.sub('', rv, 1)
                if temp != rv:
                    rv = temp
                    rv = Porter.PARTICIPLE.sub('', rv, 1)
                else:
                    temp = Porter.VERB.sub('', rv, 1)
                    if temp == rv:
                        rv = Porter.NOUN.sub('', rv, 1)
                    else:
                        rv = temp
            else:
                rv = temp

            rv = Porter.I.sub('', rv, 1)

            if re.match(Porter.DERIVATIONAL, rv):
                rv = Porter.DER.sub('', rv, 1)

            temp = Porter.P.sub('', rv, 1)
            if temp == rv:
                rv = Porter.SUPERLATIVE.sub('', rv, 1)
                rv = Porter.NN.sub(u'н', rv, 1)
            else:
                rv = temp
            word = pre+rv
        return word
    stem = staticmethod(stem)


class Imperative:

    def __init__(self) -> None:
        self.morph = pymorphy2.MorphAnalyzer()

    def is_imperative(self, sentence: str, excl: bool = False) -> bool:
        """
            На вход строка
            Если есть слово в повелительном наклонение, возвращает True
            При excl = True, вернет True, только если говорящие не включен
            в действие (иди, идите), при идем будет False
        """
        sentence = sentence.split()

        for word in sentence:
            if self.is_word_imperative(word, excl):
                return True

        return False

    def is_word_imperative(self, word: str, excl: bool = False) -> bool:
        """
            Если слово в повелительном наклонение, возвращает True
            При excl = True, вернет True, только если говорящие не включен
             в действие (иди, идите), при идем будет False
        """
        word_morph = self.morph.parse(word)
        for w_morph in word_morph:
            if w_morph.score >= 0.4:
                if 'impr' in w_morph.tag:
                    if excl:
                        if 'excl' in w_morph.tag:
                            return True
                        else:
                            continue

                    return True

                return False

        return False


class Depreciation:

    def __init__(self) -> None:
        self.morph = pymorphy2.MorphAnalyzer()
        self.stemmer = Porter()

        self.AFFECT = re.compile(
            u"(ик|ек|к|ец|иц|оск|ечк|оньк|еньк|ышк|инш|ушк|юшк)$")

    def is_depreciation(self, sentence: str) -> tuple:
        """
            Передается предложение и возвращается tuple(bool, list) есть ли уменьшительно-ласткательные слова и список таких слов
        """
        words = sentence.split()
        words_affect = []
        affect = False

        for word in words:
            if self.is_noun(word):
                word_stem = self.word_stemming(word)
                word_affect = re.search(self.AFFECT, word_stem)
                if word_affect:
                    words_affect.append(word)
                    affect = True

        return affect, list(set(words_affect))

    def is_noun(self, word) -> bool:
        words_form = self.morph.parse(word)
        for word_form in words_form:
            if word_form.score >= 0.4:
                if 'NOUN' in word_form.tag:
                    return True

        return False

    def list_stemming(self, word_list: list) -> list:
        stem_list = []

        for word in word_list:
            stem_list.append(self.word_stemming(word))

        return stem_list

    def word_stemming(self, word: str) -> str:
        return self.stemmer.stem(word)


class Toxic:

    def __init__(self, model_path: str) -> None:
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_len = 512
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def is_toxic(self, text) -> int:
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )

        out = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

        input_ids = out["input_ids"].to(self.device)
        attention_mask = out["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0)
        )

        prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]

        return True if prediction else False

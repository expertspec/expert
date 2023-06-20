import os
import pickle
import re

import eli5
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class Linear_insult_pred:
    """
    class for predicting whether a text belongs to a category, as well as for interpreting the result.
    To select a language, the class must be created with the desired language in the format "eng"/"rus".
    Returns the dictionary.
    Class category - ['verdict']
    html for interpreting - ['html']
    ====================================================================================================
    Класс для предсказания принадлежности текста к какой либо категории а так же интерпритации результата
    Для выбора языка нужно передать при создании класса нужный язык в формате "eng"/"rus"
    Возвращает словарь.
    Категория класса - ['verdict']
    html для интерпритации - ['html']
    """

    def __init__(self, lang: str = "eng"):
        self.lang = lang
        self.preobr = {
            0: "Текст без оскорблений и сарказма",
            1: "Текст с оскорблениями",
            2: "Текст с сарказмом",
        }

        with open(
            os.path.join(os.getcwd(), "text_agressive", "models", f"linear1_{lang}.pkl"),
            "rb",
        ) as f:
            self.model = pickle.load(f)

        with open(
            os.path.join(
                os.getcwd(),
                "text_agressive",
                "models",
                f"linear1_{lang}_transformer.pkl",
            ),
            "rb",
        ) as f:
            self.text_trans = pickle.load(f)

        if lang == "rus":
            # python -m spacy download ru_core_news_lg !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.russian_nlp = spacy.load("ru_core_news_lg")

    def remove_stop_and_make_lemma(
        self,
        data: str = "",
        lang: str = "ru",
    ):
        """
        Prepares the raw text for further processing

        :param data: Text to be processed, defaults to ""
        :type data: str, optional
        :param lang: The language in which the text is written, defaults to "en"
        :type lang: str, optional
        :return: Return text token list.
        :rtype: List
        ============================================================
        Подготавливает сырой текст для дальнейшей работы

        :param data: Текст, который нужно обработать, defaults to ""
        :type data: str, optional
        :param lang: Язык, на котором написан текст, defaults to "en"
        :type lang: str, optional
        :return: Возвращает лист токенов текста.
        :rtype: List
        """

        data = re.sub(r"[^a-zA-Zа-яА-Я!?,.]", " ", data)
        my_data = self.russian_nlp(str(data.lower()))
        tokens = " ".join([token.lemma_ for token in my_data if (not token.is_stop)])

        return tokens

    def predict(self, text: str = "") -> dict:
        if self.lang == "eng":
            verdict = self.model.predict(self.text_trans.transform([text.lower()]))[0]
        else:
            verdict = self.model.predict(
                self.text_trans.transform([self.remove_stop_and_make_lemma(text)])
            )[0]
        html_ipynb = eli5.show_prediction(
            self.model,
            doc=text,
            vec=self.text_trans.named_steps["vect"],
            feature_names=self.text_trans.named_steps["vect"].get_feature_names_out(),
        )
        return {"verdict": self.preobr[verdict], "html": html_ipynb.data}


if __name__ == "__main__":
    ext = Linear_insult_pred(lang="rus")

    print(
        ext.predict(
            "Я работаю 40 часов в неделю, для того чтобы оставаться бедным"
        )["verdict"]
    )

    print(ext.predict("Ну ты и тварь, конечно.")["verdict"])
    print(ext.predict("Ушлёпок. Я до тебя ещё доберусь.")["verdict"])
    print(ext.predict("А тут мы можем наблюдать восходящий тренд.")["verdict"])

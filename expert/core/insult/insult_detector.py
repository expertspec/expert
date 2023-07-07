import json
import os
import pickle
import re
from os import PathLike

import eli5
import numpy as np
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from expert.data.annotation.speech_to_text import get_sentences


def get_folder_order(path: str | PathLike) -> list:
    """Determines the path to the file entered by the user.

    Args:
        path (str | PathLike): The path to the file.

    Returns:
        list: A list where each element represents a separate directory.
    """
    folder_order = []
    while True:
        path, folder = os.path.split(path)
        if folder != "":
            folder_order.insert(0, folder)
        else:
            break
    return folder_order


class InsultDetector:
    """
    Class for predicting whether a text belongs to a category, as well as interpreting the result.
    To select a language, the class must be created with the desired language in the format "en"/"ru".

    Args:
        video_path (str | PathLike): Path to local video file.
        transcription_path (str | PathLike): Path to JSON file with text transcription.
        output_dir (str | Pathlike | None, optional): Path to the folder for saving results. Defaults to None.
        lang (str, optional): Speech language for text processing ['ru', 'en']. Defaults to 'en'.

    Returns:
        Dictionary with keys:
            - verdict: string representation of the prediction (e.g. "Text with insults")
            - probability: float prediction probability (e.g. 0.71)
            - html: html representation of the prediction (e.g. <p>Text with insults</p>)

    Attributes:
        lang (str): language selected at initialization ("en" or "ru")
        preobr (dict): dictionary representing the mapping from prediction label to prediction string
        model: trained classifier model
        text_trans: text transformer used to preprocess input data
        russian_nlp: spaCy NLP pipeline for Russian language

    Methods:
        remove_stop_and_make_lemma(data="", lang="ru"):
            Preprocesses raw text by removing stop words and lemmatizing tokens.

            Args:
                data (str): text to be processed
                lang (str): language of the input text
            Returns:
                list of processed tokens

        predict(text="") -> dict:
            Predicts the category of the input text and returns a dictionary with the prediction label and html representation.

            Args:
                text (str): input text
            Returns:
                dictionary with keys 'verdict', 'probability' and 'html'

        get_insult(self) -> str | PathLike:
            The main function to analyze the text in tandem with other modules

            Returns:
                path to the analyzed file
    """

    def __init__(
        self,
        video_path: str | PathLike,
        transcription_path: str | PathLike,
        output_dir: str | PathLike | None = None,
        lang: str = "en",
    ):
        """Initializes InsultDetector object with the specified language."""

        self.transcription_path = transcription_path

        if not output_dir:
            self.basename = os.path.splitext(os.path.basename(video_path))[0]
            print(get_folder_order(video_path))
            self.output_dir = os.path.join(*(get_folder_order(video_path)[:-1]), "temp")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        if not output_dir:
            self.output_dir = os.path.join(self.output_dir, self.basename + ".json")

        self.lang = lang
        self.preobr = {
            0: "Text without insults and sarcasm",
            1: "Text with insults",
            2: "Text with sarcasm",
        }

        # Load classifier model and text transformer
        with open(
            os.path.join(
                os.getcwd(),
                "EXPERT_NEW",
                "text_agressive",
                "models",
                f"linear_{lang}.pkl",
            ),
            "rb",
        ) as f:
            self.model = pickle.load(f)

        with open(
            os.path.join(
                os.getcwd(),
                "EXPERT_NEW",
                "text_agressive",
                "models",
                f"linear_{lang}_transformer.pkl",
            ),
            "rb",
        ) as f:
            self.text_trans = pickle.load(f)

        # Load spaCy NLP pipeline for Russian language
        if lang == "ru":
            self.russian_nlp = spacy.load("ru_core_news_lg")

    def remove_stop_and_make_lemma(
        self,
        data: str,
    ) -> list[str]:
        """Preprocesses raw text by removing stop words and lemmatizing tokens.

        Args:
            data (str): text to be processed

        Returns:
            list of processed tokens
        """

        data = re.sub(r"[^a-zA-Zа-яА-Я!?,.]", " ", data)
        my_data = self.russian_nlp(str(data.lower()))
        tokens = " ".join([token.lemma_ for token in my_data if (not token.is_stop)])

        return tokens

    def predict(self, text: str) -> dict:
        """Predicts the category of the input text and returns a dictionary with the prediction label, its probability and html representation.

        Args:
            text (str): input text

        Returns:
            dictionary with keys 'verdict', 'probability' and 'html'
        """

        if self.lang == "en":
            verdict = self.model.predict_proba(
                self.text_trans.transform([text.lower()])
            )[0]
        else:
            verdict = self.model.predict_proba(
                self.text_trans.transform([self.remove_stop_and_make_lemma(text)])
            )[0]

        # Generate html representation of the prediction
        html_ipynb = eli5.show_prediction(
            self.model,
            doc=text,
            vec=self.text_trans.named_steps["vect"],
            feature_names=self.text_trans.named_steps["vect"].get_feature_names_out(),
        )

        otv = {
            "verdict": self.preobr[np.argmax(verdict)],
            "probability": np.max(verdict),
            "html": html_ipynb.data,
        }
        # Enable if there is a high probability of false positives for the presence of insults and sarcasm

        # if np.argmax(verdict) !=0 and otv["probability"] < 0.65:
        #     otv["verdict"] = self.preobr[0]

        return otv

    def get_insult(self) -> str:  # | PathLike:
        """The main function to analyze the text in tandem with other modules"""
        insult_information = []
        with open(self.transcription_path, "r", encoding="utf-8") as f:
            data = get_sentences(json.load(f))

        for sentence in data:
            print(sentence)
            insult_information.append(
                self.predict(sentence["text"])
                | {
                    "time_start": sentence["time_start"],
                    "time_end": sentence["time_end"],
                }
            )

        with open(self.output_dir, "w", encoding="utf-8") as f:
            json.dump(insult_information, f)

        return self.output_dir


if __name__ == "__main__":
    ext = InsultDetector(
        video_path="./EXPERT_NEW/text_agressive/models/transcription.json",
        transcription_path="./EXPERT_NEW/text_agressive/models/transcription.json",
        lang="ru",
    )

    print(
        ext.predict("Я работаю 40 часов в неделю, для того чтобы оставаться бедным")[
            "verdict"
        ]
    )

    print(ext.predict("Ну ты и тварь, конечно.")["verdict"])
    print(ext.predict("Ушлёпок. Я до тебя ещё доберусь.")["verdict"])
    print(ext.predict("А тут мы можем наблюдать восходящий тренд.")["verdict"])
    ext.get_insult()

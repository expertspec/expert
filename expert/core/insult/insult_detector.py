import os
import pickle
import re

import eli5
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class InsultDetector:
    """
    Class for predicting whether a text belongs to a category, as well as interpreting the result.
    To select a language, the class must be created with the desired language in the format "eng"/"rus".
    
    Returns:
        Dictionary with keys:
            - verdict: string representation of the prediction (e.g. "Text with insults")
            - html: html representation of the prediction (e.g. <p>Text with insults</p>)
    
    Attributes:
        lang (str): language selected at initialization ("eng" or "rus")
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
                dictionary with keys 'verdict' and 'html'
    """

    def __init__(self, lang: str = "eng"):
        """Initializes InsultDetector object with the specified language."""
        
        self.lang = lang
        self.preobr = {
            0: "Text without insults and sarcasm",
            1: "Text with insults",
            2: "Text with sarcasm",
        }

        # Load classifier model and text transformer
        with open(
            os.path.join(os.getcwd(), "EXPERT_NEW", "text_agressive", "models", f"linear1_{lang}.pkl"),
            "rb",
        ) as f:
            self.model = pickle.load(f)

        with open(
            os.path.join(
                os.getcwd(),
                "EXPERT_NEW",
                "text_agressive",
                "models",
                f"linear1_{lang}_transformer.pkl",
            ),
            "rb",
        ) as f:
            self.text_trans = pickle.load(f)

        # Load spaCy NLP pipeline for Russian language
        if lang == "rus":
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
        """Predicts the category of the input text and returns a dictionary with the prediction label and html representation.
        
        Args:
            text (str): input text
        
        Returns:
            dictionary with keys 'verdict' and 'html'
        """
        
        if self.lang == "eng":
            verdict = self.model.predict(self.text_trans.transform([text.lower()]))[0]
        else:
            verdict = self.model.predict(
                self.text_trans.transform([self.remove_stop_and_make_lemma(text)])
            )[0]
        
        # Generate html representation of the prediction
        html_ipynb = eli5.show_prediction(
            self.model,
            doc=text,
            vec=self.text_trans.named_steps["vect"],
            feature_names=self.text_trans.named_steps["vect"].get_feature_names_out(),
        )
        
        return {"verdict": self.preobr[verdict], "html": html_ipynb.data}

if __name__ == "__main__":
    ext = InsultDetector(lang="rus")

    print(
        ext.predict(
            "Я работаю 40 часов в неделю, для того чтобы оставаться бедным"
        )["verdict"]
    )

    print(ext.predict("Ну ты и тварь, конечно.")["html"])
    print(ext.predict("Ушлёпок. Я до тебя ещё доберусь.")["verdict"])
    print(ext.predict("А тут мы можем наблюдать восходящий тренд.")["verdict"])

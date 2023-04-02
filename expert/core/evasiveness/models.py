from typing import List

from transformers import pipeline


def get_models(lang):
    assert lang in [
        "en",
        "ru",
    ], "Only english and russian languages are supported"
    if lang == "en":
        return EvasiveAnswers(), QuestionStatement()
    elif lang == "ru":
        return (
            EvasiveAnswers(
                model_path="AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru",
                model_limit=0.5,
            ),
            QuestionStatement(),
        )


class EvasiveAnswers:
    def __init__(
        self, model_path="deepset/roberta-base-squad2", model_limit=0.055
    ):
        self.MODEL_PATH = model_path
        self.ev_detection = pipeline(
            "question-answering", model=self.MODEL_PATH
        )
        self.limit = model_limit

    def get_evas_info(self, question: str, answer: str) -> List:
        """Classify whether an answer is evasive

        Args:
            question (str): question
            answer (str): corresponding answer

        Returns:
            List[str, int, str]: the first element is question label(not evasive, evasive, neutral),
            the second - model's confidence that found answer is the answer, the third - found answer
        """
        qa_input = {"question": question, "context": answer}
        try:
            res = self.ev_detection(qa_input)
            if res["score"] > self.limit:
                return ["not evasive", res["score"], res["answer"]]
            else:
                return ["evasive", res["score"], res["answer"]]
        except (
            Exception
        ):  # if answer length is too long, a neutral label is assigned
            return ["neutral", -1, ""]


class QuestionStatement:
    def __init__(self):
        self.MODEL_PATH = "shahrukhx01/question-vs-statement-classifier"
        self.qs_classifier = pipeline(
            "text-classification", model=self.MODEL_PATH
        )

    def get_qs_label(self, text: str) -> int:
        """Classify whether text is a question or a statement

        Args:
            text (str): text for which we determine is it a question or a statement

        Returns:
            int: label(0 - statement, 1 - question)
        """
        try:
            if self.qs_classifier(text)[0]["label"] == "LABEL_1":
                return 1
            else:
                return 0
        except (
            RuntimeError
        ):  # if sentence length is too long, is becomes a statement
            return 0

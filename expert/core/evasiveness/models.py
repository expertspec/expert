from typing import List

from transformers import pipeline


def get_models(lang):
    assert lang == "en", "Only english language is supported"
    if lang == "en":
        return EvasiveAnswers(), QuestionStatement()


class EvasiveAnswers:
    def __init__(self):
        self.MODEL_PATH = "deepset/roberta-base-squad2"
        self.ev_detection = pipeline(
            "question-answering", model=self.MODEL_PATH
        )

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
            if res["score"] > 0.055:
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

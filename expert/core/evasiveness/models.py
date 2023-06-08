from typing import List
import torch
import pickle
import numpy as np
from transformers import pipeline


def get_models(lang):
    assert lang in [
        "en",
        "ru",
    ], "Only english and russian languages are supported"
    if lang == "en":
        return EvasiveAnswers()
    elif lang == "ru":
        return EvasiveAnswers(
                qa_model_path="mrm8488/bert-multi-cased-finetuned-xquadv1",
                ev_model_path="alenaa/ru_evasiveness",
                classifier_path='rf_evas_model_ru.pickle'
            )

class EvasiveAnswers:
    def __init__(self, qa_model_path="deepset/roberta-base-squad2", ev_model_path="alenaa/evasiveness",
                 classifier_path='rf_evas_model.pickle'):
        self.QA_MODEL_PATH = qa_model_path
        self.EV_MODEL_PATH = ev_model_path
        self.answer_detection = pipeline(
            "question-answering", model=self.QA_MODEL_PATH
        )
        self.evasiveness_detection = pipeline("text-classification", model=ev_model_path)
        self.classifier = pickle.load(open(classifier_path, 'rb'))

    def get_evasive_info(self, question: str, answer: str) -> List:

        """Classify whether an answer is evasive

        Args:
            question (str): question
            answer (str): corresponding answer

        Returns:
            List[str, int, str]: the first element is question label(not evasive, evasive, neutral),
            the second - model's confidence that found answer is the answer, the third - found answer
        """

        def find_full_answer(text: str, start: int , end: int) -> str:
            """Find full answer in text

            Args:
                text (str): full text(respondent's speech part)
                start(int): response start position(according to the question answering model)
                end(int): response start position(according to the question answering model)

            Returns:
                str: full sentence with answer
            Examples:
                >>> txt = "My name is Sam. I live in London."
                >>> find_full_answer(txt, 11, 13)
                'My name is Sam'
            """
            while (start > 0) and (text[start - 1] not in ["!", "?", "."]):
                start -= 1
            while (end != len(text)) and (text[end] not in ["!", "?", "."]):
                end += 1
            return text[start:end]

        qa_input = {"question": question, "context": answer}
        try:
            answer_info = self.answer_detection(qa_input)
            full_answer = find_full_answer(answer, answer_info['start'], answer_info['end'])
            confidence = answer_info['score']
            evasive_input = question + full_answer
            evasive_info = self.evasiveness_detection(evasive_input)
            if evasive_info[0]['label'] == 'LABEL_0':
                pred_evasive = 0
            else:
                pred_evasive = 1
            cl_input = np.array([float(confidence), int(pred_evasive)]).reshape(1, -1)
            res = self.classifier.predict(cl_input)[0]

            if res == 0:
                return ["not evasive", confidence, full_answer]
            else:
                return ["evasive", confidence, full_answer]

        except(
                Exception
        ):  # if answer length is too long, a neutral label is assigned
            return ["neutral", -1, ""]


'''
class QuestionStatement:
    def __init__(self, lang):
        self.MODEL_PATH = "shahrukhx01/question-vs-statement-classifier"
        self.qs_classifier = pipeline(
            "text-classification", model=self.MODEL_PATH
        )

        self.lang = lang

    def get_qs_label(self, text: str) -> int:
        """Classify whether text is a question or a statement

        Args:
            text (str): text for which we determine is it a question or a statement

        Returns:
            int: label(0 - statement, 1 - question)
        """
        if self.lang == "ru":
            return 0
        try:
            if self.qs_classifier(text)[0]["label"] == "LABEL_1":
                return 1
            else:
                return 0
        except (
                RuntimeError
        ):  # if sentence length is too long, is becomes a statement
            return 0
'''
import pytest
import os
import json
import re

from expert.core.evasiveness.evasiveness_analysis import EvasivenessDetector
from expert.core.evasiveness.evas_models import get_models
from expert.core.evasiveness.evas_models import EvasiveAnswers

@pytest.fixture()
def evasive_model_en() -> EvasiveAnswers:
    return get_models('en')

@pytest.fixture()
def evasive_model_ru() -> EvasiveAnswers:
    return get_models('ru')


# tests for get_models from evas_models

def test_get_en_model():
    assert get_models('en').QA_MODEL_PATH == "deepset/roberta-base-squad2"
    assert get_models('en').EV_MODEL_PATH == "alenaa/evasiveness"


def test_get_ru_model():
    assert get_models('ru').QA_MODEL_PATH == "mrm8488/bert-multi-cased-finetuned-xquadv1"
    assert get_models('ru').EV_MODEL_PATH == "alenaa/ru_evasiveness"


def test_get_unknown_lang_model():
    with pytest.raises(NotImplementedError) as exc:
        get_models('de')
    assert "'lang' must be 'en' or 'ru'." in str(exc.value)


# tests for get_evasive_info from evas_models

correct_pair_en1 = {'question': 'What is your name?', 'answer': 'My name is Sam'}
correct_pair_en2 = {'question': 'How old are you?', 'answer': 'I live in London'}
correct_pair_en3 = {'question': 'How old are you?', 'answer': "I live in London. I'm 20 years old."}
incorrect_pair1 = {'question': '', 'answer': ''}
incorrect_pair2 = {'question': 10, 'answer': 20}


def test_correct_pair1_en_result(evasive_model_en, pair=correct_pair_en1):
    result = evasive_model_en.get_evasive_info(pair['question'], pair['answer'])
    assert type(result) == list
    assert len(result) == 3
    assert result[0] in ['evasive', 'not_evasive']
    assert 0 < result[1] < 1
    assert result[2] in pair['answer']


def test_correct_pair2_en_result(evasive_model_en, pair=correct_pair_en2):
    result = evasive_model_en.get_evasive_info(pair['question'], pair['answer'])
    assert type(result) == list
    assert len(result) == 3
    assert result[0] in ['evasive', 'not_evasive']
    assert 0 < result[1] < 1
    assert result[2] in pair['answer']


def test_correct_pair3_en_result(evasive_model_en, pair=correct_pair_en3):
    result = evasive_model_en.get_evasive_info(pair['question'], pair['answer'])
    assert type(result) == list
    assert len(result) == 3
    assert result[0] in ['evasive', 'not_evasive']
    assert 0 < result[1] < 1
    assert result[2] in pair['answer']


def test_incorrect_pair1_en_result(evasive_model_en, pair=incorrect_pair1):
    result = evasive_model_en.get_evasive_info(pair['question'], pair['answer'])
    assert type(result) == list
    assert len(result) == 3
    assert result[0] in ['neutral']
    assert result[1] == -1
    assert result[2] == ''

def test_incorrect_pair2_en_result(evasive_model_en, pair=incorrect_pair2):
    result = evasive_model_en.get_evasive_info(pair['question'], pair['answer'])
    assert type(result) == list
    assert len(result) == 3
    assert result[0] in ['neutral']
    assert result[1] == -1
    assert result[2] == ''

correct_pair_ru1 = {'question': 'Как тебя зовут?', 'answer': 'Меня зовут Петя'}
correct_pair_ru2 = {'question': 'Сколько тебе лет?', 'answer': 'Я живу в Питере'}
correct_pair_ru3 = {'question': 'Сколько тебе лет?', 'answer': "Я живу в Питере. Мне 20 лет."}

def test_correct_pair1_ru_result(evasive_model_ru, pair=correct_pair_ru1):
    result = evasive_model_ru.get_evasive_info(pair['question'], pair['answer'])
    assert type(result) == list
    assert len(result) == 3
    assert result[0] in ['evasive', 'not_evasive']
    assert 0 < result[1] < 1
    assert result[2] in pair['answer']


def test_correct_pair2_ru_result(evasive_model_ru, pair=correct_pair_ru2):
    result = evasive_model_ru.get_evasive_info(pair['question'], pair['answer'])
    assert type(result) == list
    assert len(result) == 3
    assert result[0] in ['evasive', 'not_evasive']
    assert 0 < result[1] < 1
    assert result[2] in pair['answer']


def test_correct_pair3_ru_result(evasive_model_ru, pair=correct_pair_ru3):
    result = evasive_model_ru.get_evasive_info(pair['question'], pair['answer'])
    assert type(result) == list
    assert len(result) == 3
    assert result[0] in ['evasive', 'not_evasive']
    assert 0 < result[1] < 1
    assert result[2] in pair['answer']


def test_incorrect_pair1_ru_result(evasive_model_ru, pair=incorrect_pair1):
    result = evasive_model_ru.get_evasive_info(pair['question'], pair['answer'])
    assert type(result) == list
    assert len(result) == 3
    assert result[0] in ['neutral']
    assert result[1] == -1
    assert result[2] == ''

def test_incorrect_pair2_ru_result(evasive_model_ru, pair=incorrect_pair2):
    result = evasive_model_ru.get_evasive_info(pair['question'], pair['answer'])
    assert type(result) == list
    assert len(result) == 3
    assert result[0] in ['neutral']
    assert result[1] == -1
    assert result[2] == ''


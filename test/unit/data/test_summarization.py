import pytest

from expert.data.annotation.summarization import SummarizationEN, SummarizationRU


@pytest.fixture()
def summ_ru():
    summ_ru = SummarizationRU()
    
    return summ_ru


@pytest.fixture()
def summ_en():
    summ_en = SummarizationEN()
    
    return summ_en


@pytest.fixture()
def ru_text():
    ru_text = "Cъешь ещё этих мягких французских булок, да выпей чаю"
    
    return ru_text


@pytest.fixture()
def en_text():
    en_text = "The quick brown fox jumps over the lazy dog"
    
    return en_text


def test_ru_lower_register(summ_ru, ru_text):
    pred = summ_ru._to_lower_register(ru_text)
    
    assert type(pred) == str
    assert len(pred) > 0


def test_ru_preprocess(summ_ru, ru_text):
    pred = summ_ru._preprocess(ru_text)
    
    assert type(pred) == str
    assert len(pred) > 0


def test_ru_get_summary(summ_ru, ru_text):
    pred = summ_ru.get_summary(ru_text)
    
    assert type(pred) == str
    assert len(pred) > 0


def test_en_lower_register(summ_en, en_text):
    pred = summ_en._to_lower_register(en_text)
    
    assert type(pred) == str
    assert len(pred) > 0


def test_en_preprocess(summ_en, en_text):
    pred = summ_en._preprocess(en_text)
    
    assert type(pred) == str
    assert len(pred) > 0


def test_en_get_summary(summ_en, en_text):
    pred = summ_en.get_summary(en_text)
    
    assert type(pred) == str
    assert len(pred) > 0

import pytest

from expert.core.congruence.text_emotions.text_model import TextModel


@pytest.fixture()
def ru_model():
    model = TextModel(lang="ru")

    return model


@pytest.fixture()
def en_model():
    model = TextModel(lang="en")

    return model


@pytest.fixture()
def ru_text():
    text = "Cъешь ещё этих мягких французских булок, да выпей чаю"

    return text


@pytest.fixture()
def en_text():
    text = "The quick brown fox jumps over the lazy dog"

    return text


def test_ru_text_model(ru_model, ru_text):
    pred = ru_model.predict(text=ru_text)

    assert type(pred) == dict
    assert len(list(pred.values())) == 3
    assert pred["anger"] == pred["anger"]
    assert pred["neutral"] == pred["neutral"]
    assert pred["happiness"] == pred["happiness"]
    assert sum(list(pred.values())) >= 0


def test_en_text_model(en_model, en_text):
    pred = en_model.predict(text=en_text)

    assert type(pred) == dict
    assert len(list(pred.values())) == 3
    assert pred["anger"] == pred["anger"]
    assert pred["neutral"] == pred["neutral"]
    assert pred["happiness"] == pred["happiness"]
    assert sum(list(pred.values())) >= 0

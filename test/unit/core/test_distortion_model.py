import pytest

from expert.core.distortion.dist_model import DistortionModel


@pytest.fixture()
def ru_model():
    model = DistortionModel(lang="ru")

    return model


@pytest.fixture()
def en_model():
    model = DistortionModel(lang="en")
    
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

    assert type(pred) == list
    assert len(pred) == 8
    assert pred[0]["score"] == pred[0]["score"]
    assert pred[1]["score"] == pred[1]["score"]
    assert pred[2]["score"] == pred[2]["score"]
    assert pred[3]["score"] == pred[3]["score"]
    assert pred[4]["score"] == pred[4]["score"]
    assert pred[5]["score"] == pred[5]["score"]
    assert pred[6]["score"] == pred[6]["score"]
    assert pred[7]["score"] == pred[7]["score"]
    assert sum([p["score"] for p in pred]) >= 0


def test_en_text_model(en_model, en_text):
    pred = en_model.predict(text=en_text)

    assert type(pred) == list
    assert len(pred) == 8
    assert pred[0]["score"] == pred[0]["score"]
    assert pred[1]["score"] == pred[1]["score"]
    assert pred[2]["score"] == pred[2]["score"]
    assert pred[3]["score"] == pred[3]["score"]
    assert pred[4]["score"] == pred[4]["score"]
    assert pred[5]["score"] == pred[5]["score"]
    assert pred[6]["score"] == pred[6]["score"]
    assert pred[7]["score"] == pred[7]["score"]
    assert sum([p["score"] for p in pred]) >= 0


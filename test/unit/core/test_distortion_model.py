import pytest

from expert.core.distortion.dist_model import DistortionModel
from expert.core.distortion.distortion_analysis import DistortionDetector
import torch
import json
import os


@pytest.fixture()
def dist_detector():
    dist_detector = DistortionDetector(
        video_path="test/assets/videos/dialogue.mp4",
        features_path="test/assets/videos/dialogue/features.json",
        face_image="test/assets/videos/dialogue/0.jpg",
        transcription_path="test/assets/videos/dialogue/transcription.json",
        diarization_path="test/assets/videos/dialogue/diarization.json",
        device=torch.device("cpu"),
    )

    return dist_detector


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


def test_dist_analysis(dist_detector):
    div_distortions_path, agg_distortions_path = dist_detector.get_distortion()

    assert os.path.isfile(div_distortions_path)
    assert os.path.isfile(agg_distortions_path)

    with open(div_distortions_path, "r") as file:
        div_distortions = json.load(file)

    assert type(div_distortions) == list
    assert len(div_distortions) > 0

    with open(agg_distortions_path, "r") as file:
        pred = json.load(file)

    assert type(pred) == list
    assert len(pred) == 1
    assert pred[0]["NO DISTORTION"] == pred[0]["NO DISTORTION"]
    assert pred[0]["SHOULD STATEMENTS"] == pred[0]["SHOULD STATEMENTS"]
    assert pred[0]["LABELING"] == pred[0]["LABELING"]
    assert pred[0]["REWARD FALLACY"] == pred[0]["REWARD FALLACY"]
    assert pred[0]["CATASTROPHIZING"] == pred[0]["CATASTROPHIZING"]
    assert pred[0]["OVERGENERALIZING"] == pred[0]["OVERGENERALIZING"]
    assert pred[0]["PERSONALIZATION"] == pred[0]["PERSONALIZATION"]
    assert pred[0]["EMOTIONAL REASONING"] == pred[0]["EMOTIONAL REASONING"]
    assert sum(list(pred[0].values())) >= 0

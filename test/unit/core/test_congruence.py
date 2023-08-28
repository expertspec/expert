import pytest

import timm
from timm.models.layers import conv2d_same
from expert.core.congruence.congruence_analysis import CongruenceDetector
import json
import os


@pytest.fixture()
def cong_detector():
    cong_detector = CongruenceDetector(
        video_path="test/assets/videos/dialogue.mp4",
        features_path="test/assets/videos/dialogue/features.json",
        face_image="test/assets/videos/dialogue/0.jpg",
        transcription_path="test/assets/videos/dialogue/transcription.json",
        diarization_path="test/assets/videos/dialogue/diarization.json",
    )

    return cong_detector


def test_cong(cong_detector):
    emotions_path, congruence_path = cong_detector.get_congruence()

    assert os.path.isfile(emotions_path)
    assert os.path.isfile(congruence_path)

    with open(emotions_path, "r") as file:
        emotions = json.load(file)

    assert type(emotions) == dict
    assert type(emotions["video"]) == list
    assert len(emotions["video"]) > 0
    assert type(emotions["audio"]) == list
    assert len(emotions["audio"]) > 0
    assert type(emotions["text"]) == list
    assert len(emotions["text"]) > 0

    with open(congruence_path, "r") as file:
        congruence = json.load(file)

    assert type(congruence) == list
    assert len(congruence) > 0

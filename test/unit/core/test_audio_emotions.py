import pytest

import timm
from timm.models.layers import conv2d_same
from expert.core.congruence.audio_emotions.audio_analysis import AudioAnalysis


@pytest.fixture()
def model():
    model = AudioAnalysis(
        video_path="test/assets/test_audio.mp3",
        stamps={"SPEAKER_00": [[0, 10]]},
    )

    return model


def test_audio(model):
    pred = model.predict()

    assert type(pred) == list
    assert type(pred[0]) == dict
    assert len(pred[0].values()) == 4
    assert pred[0]["audio_anger"] == pred[0]["audio_anger"]
    assert pred[0]["audio_neutral"] == pred[0]["audio_neutral"]
    assert pred[0]["audio_happiness"] == pred[0]["audio_happiness"]
    assert sum(list(pred[0].values())) >= 0

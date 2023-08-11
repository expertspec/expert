import os

import pytest

from expert.data.diarization import audio_transforms
from expert.data.diarization import speaker_diarization


@pytest.fixture()
def timestamps():
    timestamps = {
        "SPEAKER_00": [[0.3, 14.51], [39.41, 42.01]],
        "SPEAKER_01": [[14, 39]],
    }

    return timestamps


def test_get_rounded_intervals(timestamps):
    rounded_stamps = audio_transforms.get_rounded_intervals(timestamps)

    assert type(rounded_stamps) == dict
    assert rounded_stamps == {
        "SPEAKER_00": [[0, 15], [39, 43]],
        "SPEAKER_01": [[14, 39]],
    }

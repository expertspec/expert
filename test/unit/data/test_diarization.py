import os

import pytest
from expert.data.diarization.speaker_diarization import SpeakerDiarization


def test_diarization():
    diarization = SpeakerDiarization(audio="./test/assets/videos/dialogue.mp4")
    stamps = diarization.apply()

    assert type(stamps) == dict
    assert len(stamps) == 2
    assert type(stamps["SPEAKER_00"][0]) == list
    assert type(stamps["SPEAKER_00"][0][0]) == int

import os

import pytest
from expert.data.diarization.speaker_diarization import SpeakerDiarization


def test_diarization():
    diarization = SpeakerDiarization(audio="./test/assets/test_audio.mp3")
    stamps = diarization.apply()

    assert type(stamps) == dict
    assert len(stamps) == 1
    assert type(stamps["SPEAKER_00"][0]) == list
    assert type(stamps["SPEAKER_00"][0][0]) == int

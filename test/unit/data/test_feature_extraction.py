import os

import pytest

from expert.data.feature_extractor import FeatureExtractor


@pytest.fixture()
def extractor():
    extractor = FeatureExtractor(video_path="test/assets/videos/dialogue.mp4")
    return extractor


def test_output_path(extractor):
    assert "dialogue" in extractor.temp_path


def test_duration(extractor):
    assert extractor.phrase_duration > 0


def test_lang(extractor):
    assert isinstance(extractor.lang, str)

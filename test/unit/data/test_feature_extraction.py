import os

import pytest

from expert.data.feature_extractor import FeatureExtractor


@pytest.fixture()
def extractor():
    extractor = FeatureExtractor(video_path="test/data/video/dialogue1.mp4")
    return extractor

def test_path(extractor):
    assert "/" in extractor.video_path

def test_duration(extractor):
    assert extractor.phrase_duration > 0

def test_lang(extractor):
    assert isinstance(extractor.lang, str)
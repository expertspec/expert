import pytest

from expert.core.aggression.aggression_analysis import AggressionDetector
import torch
import json
import os


@pytest.fixture()
def detector():
    detector = AggressionDetector(
        video_path="test/assets/videos/dialogue.mp4",
        features_path="test/assets/videos/dialogue/features.json",
        face_image="test/assets/videos/dialogue/0.jpg",
        transcription_path="test/assets/videos/dialogue/transcription.json",
        diarization_path="test/assets/videos/dialogue/diarization.json",
        lang="en",
        device=torch.device("cpu"),
    )

    return detector


@pytest.fixture()
def fragments():
    fragments = [
        {"time_sec": 0, "text": "The quick brown fox jumps over the lazy dog"},
        {"time_sec": 10, "text": "The quick brown dog jumps over the lazy fox"},
    ]

    return fragments


def test_video_analysis(detector):
    div_vid_agg, full_vid_agg, key = detector.get_video_state()

    assert type(div_vid_agg) == list
    assert len(div_vid_agg) > 0
    assert type(full_vid_agg) == dict
    assert full_vid_agg["uppface_activity"] == full_vid_agg["uppface_activity"]
    assert (
        full_vid_agg["low_face_activity"] == full_vid_agg["low_face_activity"]
    )
    assert full_vid_agg["rapid_activity"] == full_vid_agg["rapid_activity"]
    assert full_vid_agg["rotate_activity"] == full_vid_agg["rotate_activity"]
    assert type(key) == str


def test_audio_analysis(detector):
    div_aud_agg, full_aud_agg = detector.get_audio_state(
        stamps=detector.stamps["SPEAKER_00"]
    )

    assert type(div_aud_agg) == list
    assert len(div_aud_agg) > 0
    assert type(full_aud_agg) == dict
    assert full_aud_agg["loud_part"] == full_aud_agg["loud_part"]
    assert full_aud_agg["fast_part"] == full_aud_agg["fast_part"]


def test_text_analysis(detector, fragments):
    div_text_agg, full_text_agg = detector.get_text_state(fragments=fragments)

    assert type(div_text_agg) == list
    assert len(div_text_agg) > 0
    assert type(full_text_agg) == dict
    assert full_text_agg["toxic_part"] == full_text_agg["toxic_part"]
    assert (
        full_text_agg["depreciation_part"] == full_text_agg["depreciation_part"]
    )
    assert full_text_agg["imperative_part"] == full_text_agg["imperative_part"]


def test_anger_analysis(detector):
    div_agg_path, full_agg_path = detector.get_aggression()

    assert os.path.isfile(div_agg_path)
    assert os.path.isfile(full_agg_path)

    with open(div_agg_path, "r") as file:
        div_agg = json.load(file)

    assert type(div_agg) == dict
    assert type(div_agg["video"]) == list
    assert len(div_agg["video"]) > 0
    assert type(div_agg["audio"]) == list
    assert len(div_agg["audio"]) > 0
    assert type(div_agg["text"]) == list
    assert len(div_agg["text"]) > 0

    with open(full_agg_path, "r") as file:
        full_agg = json.load(file)

    assert type(full_agg) == dict
    assert type(full_agg["video"]) == dict
    assert type(full_agg["audio"]) == dict
    assert type(full_agg["text"]) == dict

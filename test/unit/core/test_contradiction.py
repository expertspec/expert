import pytest
import torch
import json
from expert.core.contradiction import contradiction_analysis
from expert.data.annotation.speech_to_text import get_phrases


def get_contr_detector():
    detector = contradiction_analysis.ContradictionDetector(transcription_path="./test/assets/dialogue_transcription.json",
                                                            video_path="./test/assets/videos/dialogue.mp4")
    
    return detector

def test_get_utterances():
    detector = get_contr_detector() 
    
    with open(detector.transcription_path, "r") as f:
        words = json.load(f)
    
    texts = detector.get_utterances(words)
    
    assert len(texts) > 0
    assert type(texts[0]) == dict
    
def test_analysis():
    detector = get_contr_detector()
    
    result = detector.analysis(entered_text="there are some cats",
                               utterances={"text": "there areplenty of dogs",
                                           "time_interval": (0, 1)})
    assert type(result) == list
    assert result[0]["predict"] == 1.0
from __future__ import annotations

import torch
from torch import nn
from typing import List
import pandas as pd
import json

from app.libs.expert.expert.data.annotation.speech_to_text import get_phrases
from app.libs.expert.expert.core.congruence.text_emotions.text_model import TextModel


def get_text_fragments(words_path: str, stamps: dict, key: str) -> List:
    """Getting text fragments for selected expert from transcription with a window of 10 seconds.
    
    Args:
        words_path (str): Path to JSON file with text transcription.
        stamps (dict): Dictionary with diarization information.
        key (str): Expert selected by user.
    """
    
    with open(words_path, "r") as file:
        words = json.load(file)
    
    phrases = get_phrases(words, duration=10)
    data = pd.DataFrame(data=phrases)
    fragments = []
    
    for start_sec, finish_sec in stamps[key]:
        for row in range(len(data)):
            if data["time"][row][0] > start_sec-5 and data["time"][row][1] < finish_sec+5:
                fragments.append({
                    "time_sec": float(data["time"][row][0] - data["time"][row][0] % 10),
                    "text": data["text"][row]
                })
    
    return fragments


def get_text_emotions(words_path: str, stamps: str, key: str, device: torch.device | None = None) -> List:
    """Classification of expert emotions in text.
    
    Args:
        words_path (str): Path to JSON file with text transcription.
        stamps (str): Dictionary with diarization information.
        key (str): Expert selected by user.
    """
    softmax = nn.Softmax(dim=1)
    emo_model = TextModel(device=device)
    fragments = get_text_fragments(words_path, stamps, key)
    data = pd.DataFrame(data=fragments)
    
    for row in range(len(data)):
        emotion_dict = emo_model.predict(data["text"][row])
        lim_emotions = softmax(torch.Tensor([[
            emotion_dict["anger"],
            emotion_dict["neutral"],
            emotion_dict["happiness"]
        ]]))[0].numpy()
        data.loc[row, "text_anger"] = float(lim_emotions[0])
        data.loc[row, "text_neutral"] = float(lim_emotions[1])
        data.loc[row, "text_happiness"] = float(lim_emotions[2])
    
    return data.drop(["text"], axis=1).to_dict("records")
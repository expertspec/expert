from __future__ import annotations

import json
import subprocess
from vosk import KaldiRecognizer, Model, SetLogLevel
from typing import List
from os import PathLike

SetLogLevel(-1)


def transcribe_video(video_path: str | PathLike, sample_rate: int = 16000) -> List:
    """Speech recognition module from video.
    
    Args:
        video_path (str | Pathlike): Path to local video file.
        sample_rate (int, optional): Sample rate. Defaults to 16000.
    """
    model = Model(lang="en-us")
    rec = KaldiRecognizer(model, sample_rate)
    rec.SetWords(True)
    
    command = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "quiet",
        "-i",
        video_path,
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-f",
        "s16le",
        "-",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    
    results = []
    while True:
        data = process.stdout.read(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            results.append(rec.Result())
    results.append(rec.FinalResult())
    
    return results


def get_all_words(transcribation: List) -> List:
    """Get all stamps with words from the transcribed text.
    
    Args:
        transcribation(List): Speech recognition module result.
    """
    all_words = []
    for i, res in enumerate(transcribation):
        words = json.loads(res).get("result")
        if not words:
            continue
        for w in words:
            all_words.append(w)
    
    return all_words


def get_phrases(all_words: List, duration: int = 10) -> List:
    """Split transcribed text into segments of a fixed length.
    
    Args:
        all_words (List): All stamps with words from the transcribed text.
        duration (int, optional): Length of intervals for extracting phrases from speech. Defaults to 10.
    """
    phrases = []
    
    assert len(all_words) > 1, "Not enough words in text."
    
    while all_words:
        init_elem = all_words.pop(0)
        phrase = init_elem["word"]
        time_left = duration - (init_elem["end"] - init_elem["start"])
        end_time = init_elem["end"]
        while time_left > 0 and all_words:
            elem = all_words.pop(0)
            phrase = phrase + " " + elem["word"]
            time_left -= elem["end"] - end_time
            end_time = elem["end"]
        else:
            phrases.append({"time": [init_elem["start"], elem["end"]], "text": phrase})
    
    return phrases
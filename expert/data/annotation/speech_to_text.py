from __future__ import annotations

import torch
import whisper
from typing import List, Dict
from os import PathLike

import expert.data.annotation.transcribe as transcribe


def transcribe_video(
    video_path: str | PathLike,
    lang: str = "en",
    model: str = "server",
    device: torch.device | None = None,
) -> Dict:
    """Speech recognition module from video.

    Args:
        video_path (str | Pathlike): Path to the local video file.
        lang (str, optional): Language for speech recognition ['ru', 'en']. Defaults to 'en'.
        model (str, optional): Model configuration for speech recognition ['server', 'local']. Defaults to 'server'.
        device (torch.device | None, optional): Device type on local machine (GPU recommended). Defaults to None.

    Raises:
        NotImplementedError: If 'lang' is not equal to 'en' or 'ru'.
        NotImplementedError: If 'model' is not equal to 'server' or 'local'.
    """
    if lang not in ["en", "ru"]:
        raise NotImplementedError("'lang' must be 'en' or 'ru'.")
    if model not in ["server", "local"]:
        raise NotImplementedError("'model' must be 'server' or 'local'.")

    _device = torch.device("cpu")
    if device is not None:
        _device = device

    if model == "server":
        model = whisper.load_model("medium", device=_device)
    elif model == "local":
        model = whisper.load_model("base", device=_device)

    transribation = transcribe.transcribe_timestamped(
        model=model,
        audio=video_path,
        language=lang
    )

    return transribation


def get_all_words(transcribation: Dict) -> Tuple[List, str]:
    """Get all stamps with words from the transcribed text.

    Args:
        transcribation (Dict): Speech recognition module results.
    """
    full_text = transcribation["text"]
    all_words = []
    for segment in transcribation["segments"]:
        for word in segment["words"]:
            all_words.append(word)

    return all_words, full_text


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
        phrase = init_elem["text"]
        time_left = duration - (init_elem["end"] - init_elem["start"])
        end_time = init_elem["end"]
        while time_left > 0 and all_words:
            elem = all_words.pop(0)
            phrase = phrase + " " + elem["text"]
            time_left -= elem["end"] - end_time
            end_time = elem["end"]
        else:
            phrases.append(
                {"time": [init_elem["start"], elem["end"]], "text": phrase})

    return phrases

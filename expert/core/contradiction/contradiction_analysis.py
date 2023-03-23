from __future__ import annotations

import json
import os
from datetime import timedelta
from os import PathLike
from typing import List, Tuple, Union

import torch

from expert.core.contradiction.contr_tools import model_tools
from expert.data.annotation.speech_to_text import get_phrases


class ContradictionDetector:
    """Class for working with detector of contradiction."""

    def __init__(
        self,
        transcription_path: str | PathLike,
        video_path: str | PathLike,
        lang: str = "en",
        device: torch.device | None = None,
        output_dir: str | PathLike = None,
        chosen_intervals: List[int] = None,
        interval_duration: int = 60,
    ) -> None:
        """Object initialization.
         Creates instance with information about language and device.

        Args:
            transcription_path (str | PathLike): Path to JSON file with words from transcribation.
            video_path (str | PathLike): Path to local video file.
            lang (str, optional): Speech language for text processing ['ru', 'en']. Defaults to 'en'.
            device (str, optional): Device type on local machine (GPU recommended). Defaults to None.
            output_dir (str | PathLike): Path to save JSON with information about contradictions.
            chosen_intervals (list[int], optional): Flags for intervals for analyzis. Defaults to [].
                For default settings will be analyzed full text.
            interval_duration (int, optional): The length of intervals (value in seconds). Defaults to 60.
        """
        if lang not in ["en", "ru"]:
            raise NotImplementedError("'lang' must be 'en' or 'ru'.")
        self.lang = lang

        self._device = torch.device("cpu")
        self.model = model_tools.create_model(self.lang, self._device)

        if device is not None:
            self._device = device
            self.model.to(self._device)

        self.transcription_path = transcription_path
        self.chosen_intervals = chosen_intervals
        self.interval_duration = interval_duration

        if output_dir is not None:
            self.temp_path = output_dir
        else:
            basename = os.path.splitext(os.path.basename(video_path))[0]
            self.temp_path = os.path.join("temp", basename)
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

    def get_utterances(self, all_words: List[dict]) -> List[str]:
        """
        Function for getting utterances for comparison.
        Split text with speech pauses
        Args:
            all_words (List[dict]): Fragment's words to create utterances

        Returns:
            List[str]: Utterances for comparison
        """
        utterances = []
        if all_words:
            end = all_words[0]["end"]
            utterance = all_words[0]["text"]
            start = end
            for idx in range(1, len(all_words[:-1])):
                if all_words[idx + 1]["start"] - end < 1.0:
                    utterance += " " + all_words[idx + 1]["text"]
                    end = all_words[idx + 1]["end"]
                else:
                    utterances.append(
                        {
                            "text": utterance,
                            "time_interval": (int(start), int(end)),
                        }
                    )
                    end = all_words[idx + 1]["end"]
                    start = all_words[idx + 1]["start"]
                    utterance = all_words[idx + 1]["text"]
            utterances.append(
                {"text": utterance, "time_interval": (int(start), int(end))}
            )
        else:
            raise "No words."
        return utterances

    def analysis(
        self, entered_text: str, utterances: Union[str, List[str]], ind=0
    ) -> List[dict]:
        analysis_results = []
        if isinstance(utterances, str):
            part, predict = model_tools.predict_inference(
                entered_text,
                utterances["text"],
                self.model,
                lang=self.lang,
                device=self._device,
            )
            analysis_results.append(
                {
                    "time_interval": f"{timedelta(seconds=utterances['time_interval'][0])}-{timedelta(seconds=utterances['time_interval'][1])}",
                    "text": part.replace(" [SEP]", ""),
                    "predict": float(predict),
                }
            )
        elif isinstance(utterances, list):
            for utterance in utterances:
                part, predict = model_tools.predict_inference(
                    entered_text,
                    utterance["text"],
                    self.model,
                    lang=self.lang,
                    device=self._device,
                )
                try:
                    analysis_results.append(
                        {
                            "time_interval": f"{timedelta(seconds=utterance['time_interval'][0])}-{timedelta(seconds=utterance['time_interval'][1])}",
                            "text": part.replace(" [SEP]", ""),
                            "predict": float(predict),
                        }
                    )
                except AttributeError:
                    analysis_results.append(
                        {
                            "time_interval": f"{timedelta(seconds=utterance['time_interval'][0])}-{timedelta(seconds=utterance['time_interval'][1])}",
                            "text": part[0].replace(" [SEP]", ""),
                            "predict": float(predict),
                        }
                    )
        else:
            raise TypeError
        return analysis_results

    @property
    def device(self) -> torch.device:
        """Check the device type.

        Returns:
            torch.device: Device type on local machine.
        """
        return self._device

    def get_contradiction(
        self,
        entered_text: str,
    ) -> str:
        """
        Function for text analyzing.
        Creates json file with predictions.

        Args:
            entered_text (str): Utterance for analysis.
        """
        with open(self.transcription_path, "r") as f:
            words = json.load(f)

        if self.chosen_intervals:
            fragments = self.get_phrases(
                words[:], duration=self.interval_duration
            )
            intervals = dict.fromkeys(range(len(fragments)))
            for interval in self.chosen_intervals:
                interval_words = []
                for word in words:
                    if (
                        fragments[interval]["time"][0] <= word["start"]
                        and word["end"] <= fragments[interval]["time"][1]
                    ):
                        interval_words.append(word)
                texts = self.get_utterances(interval_words)
                intervals[interval] = texts

            contr_data = []

            for ind in intervals:
                if intervals[ind]:
                    analysis_results = self.analysis(
                        entered_text, texts, ind=ind
                    )
                    contr_data.append(analysis_results)

            # Unpacking of the list of lists.
            contr_data = [item for sublist in contr_data for item in sublist]

        else:
            texts = self.get_utterances(words)
            contr_data = self.analysis(entered_text, texts)

        with open(
            os.path.join(self.temp_path, "contradiction.json"), "w"
        ) as filename:
            json.dump(contr_data, filename)

        return os.path.join(self.temp_path, "contradiction.json")

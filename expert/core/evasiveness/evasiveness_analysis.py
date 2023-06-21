from __future__ import annotations

import json
import os
import re
from os import PathLike
from typing import List, Tuple

import torch

from expert.core.evasiveness.models import get_models
from expert.data.annotation.speech_to_text import between_timestamps


class EvasivenessDetector:
    """Class for working with detector of evasiveness

    Args:
        video_path (str | PathLike): Path to the local video file.
        transcription_path (str | PathLike): Path to the result of speech recognition module.
        diarization_path (str | PathLike): Path to the result of diarization module.
        lang (str, optional): Speech language for text processing ['ru', 'en']. Defaults to 'en'.
        device (torch.device | None, optional): Device type on local machine (GPU recommended). Defaults to None.
        output_dir (str | Pathlike | None, optional): Path to the folder for saving results. Defaults to None.

    Returns:
        str: Path to JSON file with information about evasiveness. JSON includes List of lists, where the first List
        includes summary for each respondent's responses and has the following structure:
        [{"resp": speaker's name, "total": total number of questions, "evasive": number of evasive answers,
        "not evasive": number of not evasive answers, "neutral": number of neutral answers}],
        index numbers of the following lists correspond to a respondents numbers + 1. Every list consisted of
        dicts with the following structure:
        {resp: str, question: str, answer: str, label: str, model_conf: float, pred_answer: str}

    Example:
        >>> detector = EvasivenessDetector(
        >>>     video_path="test_video.mp4",
        >>>     transcription_path="temp/test_video/transcription.json",
        >>>     diarization_path="temp/test_video/diarization.json",
        >>>     lang="ru",
        >>>     device=torch.device("cuda")
        >>> )
        >>> detector.get_evasiveness()
        "temp/test_video/evasiveness.json"
    """

    def __init__(
        self,
        video_path: str | PathLike,
        transcription_path: str | PathLike,
        diarization_path: str | PathLike,
        lang: str = "en",
        device: torch.device | None = None,
        output_dir: str | PathLike | None = None,
    ) -> None:
        self.transcription_path = transcription_path
        self.diarization_path = diarization_path
        self.lang = lang

        if output_dir is not None:
            self.temp_path = output_dir
        else:
            basename = os.path.splitext(os.path.basename(video_path))[0]
            self.temp_path = os.path.join("temp", basename)
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        self._device = torch.device("cpu")
        if device is not None:
            self._device = device

        self.ev_model = get_models(self.lang)

        with open(self.diarization_path, "r", encoding="utf-8") as file:
            self.stamps = json.load(file)

        with open(self.transcription_path, "r", encoding="utf-8") as file:
            self.transcription = json.load(file)

    @property
    def device(self) -> torch.device:
        """Check the device type.

        Returns:
            torch.device: Device type on local machine.
        """
        return self._device

    def get_dialogues_phrases(self) -> Tuple[List, List]:
        """Get all phrases for all speaker

        Returns:
            Tuple[List, List]: The first List consists of tuples(sorted by timestamps), every tuple has the
            following structure: Tuple[List[int, int], str, str], where the first element - begin/end timestamps
            for a speech part, the second - speaker's name, the third - speech part
        """
        wrds = self.transcription
        speakers_names = list(self.stamps.keys())
        assert (
            len(speakers_names) > 1
        ), f"More than one speaker is needed, only {len(speakers_names)} is detected"

        def get_speaker_timestamps(speaker: str) -> List:
            """Get all phrases for a chosen speaker

            Returns:
                List: List consisted of tuples: Tuple[List[int, int], str], first element - begin/end timestamps
                for a speech part, second - speaker's name

            Example:
                >>> self.stamps = {'SPEAKER_00': [[1,2], [7,10]], 'SPEAKER_01': [[3,7], [11,15]]}
                >>> get_speaker_timestamps('SPEAKER_00')
                [([1,2], 'SPEAKER_00'), ([7,10], 'SPEAKER_00')]
            """
            sp_list = []
            speaker_stamps = self.stamps[speaker]
            for stamp in speaker_stamps:
                sp_list.append((stamp, speaker))
            return sp_list

        def merge_dialogue_timestamps(timestamps: List) -> List:
            """Merge phrases owned by the same speaker

            Args:
                timestamps (List): List consisted of tuples(sorted by timestamps), every tuple has the
                following structure: Tuple[List[int, int], str], where str = "SPEAKER_XX"(XX - speaker's number),
                List[0] - timestamp of the begging of speaker's speech part, List[1] - timestamp of the end

            Returns:
                List: List with the same structure, as timestamps

            Example:
                >>> merge_dialogue_timestamps([([5,10], 'SPEAKER_01'), ([4,5], 'SPEAKER_00'), ([1,2], 'SPEAKER_00')])
                 [([1,5], 'SPEAKER_00'), ([5,10], 'SPEAKER_01')]
            """
            merged_dialogue_timestamps = []
            i = len(timestamps) - 1
            while i >= 0:
                start = timestamps[i][0][0]
                while (i > 0) and (timestamps[i][1] == timestamps[i - 1][1]):
                    i -= 1
                end = timestamps[i][0][1]
                merged_dialogue_timestamps.append(
                    ([start, end], timestamps[i][1])
                )
                i -= 1
            return merged_dialogue_timestamps

        dialogue_timestamps = []
        for name in speakers_names:
            dialogue_timestamps += get_speaker_timestamps(name)
        dialogue_timestamps = sorted(dialogue_timestamps, reverse=True)
        dialogue_timestamps = merge_dialogue_timestamps(dialogue_timestamps)

        def cut_phrases(phrase):
            """Cut incomplete sentences

            Args:
                phrase (str): Speech part

            Returns:
                str: Cleaned speech part

            Example:
                >>> cut_phrase("tonight. I'll wait for you. When")
                 'I'll wait for you.'
            """
            pattern = r"[A-Z|А-Я].*[.?!]"
            cut_phrase = re.findall(pattern, phrase)
            if len(cut_phrase) == 0:
                return ""
            else:
                return cut_phrase[0]

        dialogue = []
        prev_text = ""
        for i in dialogue_timestamps:
            if i[0][1] > wrds[-1]["end"]:
                break
            text = between_timestamps(wrds, i[0][0], i[0][1])
            text = cut_phrases(text)
            if text == "":
                continue
            if (
                prev_text != ""
            ):  # additional check for cases, like: "USA? What do you think about it?"
                first_punc = re.search("[?!.]", text).span()[1]
                if text[:first_punc] == prev_text[-first_punc:]:
                    text = text[first_punc + 1 :]
            prev_text = text
            dialogue.append(i + (text,))

        return dialogue, speakers_names

    def get_evasiveness(self) -> str:
        dialogue, speakers_names = self.get_dialogues_phrases()
        answers = [[] for _ in range(len(speakers_names))]
        print(dialogue)

        def get_questions(text: str) -> List:
            """Find questions in full speech part

            Args:
                text (str): Speech part, where question was detected(by question/statement model or '?' was found)

            Returns:
                List: List of separated questions, detected from the speech part

            Examples:
                >>> txt = "Let's introduce ourselves. My name is Sam. What is your name?"
                >>> get_questions(txt)
                ['What is your name?']
                >>> txt = "My name is Sam. What is your name? How old are you"
                >>> get_questions(txt)
                ['What is your name?', 'How old are you']
                >>> txt = "Your father passed away last night. And I wonder why are you here?"
                >>> get_questions(txt)

                ["Your father passed away last night. And I wonder why are you here?"]
            """
            full_question = text
            text = re.findall(r"[^.!?)]+[!?.]", text)
            ques_text = []
            for sent in text:
                # if self.qs_model.get_qs_label(sent) == 1 or "?" in sent:
                if "?" in sent:
                    ques_text.append(sent)
            if len(ques_text) == 0:
                return [full_question]
            else:
                return ques_text

        def get_number(name: str) -> int:
            """Turn speaker's name into speaker's number

            Args:
                name (str): speaker's name

            Returns:
                int: speaker's number

            Example:
            >>> get_number('SPEAKER_101')
            101
            """
            num = name.split("_")[1]
            if bool(re.fullmatch("0+", num)):
                return 0

            else:
                return int(num.lstrip("0"))

        res = []  # summary of each respondent's responses
        for name in speakers_names:
            res.append(
                {
                    "resp": name,
                    "total": 0,
                    "evasive": 0,
                    "not evasive": 0,
                    "neutral": 0,
                }
            )

        for i in range(len(dialogue) - 1):
            if "?" in dialogue[i][2]:
                # or self.qs_model.get_qs_label(dialogue[i][2]) == 1

                questions = get_questions(dialogue[i][2])
                for question in questions:
                    try:
                        answer_info = self.ev_model.get_evasive_info(
                            question, dialogue[i + 1][2]
                        )
                    except Exception:
                        break
                    speaker_num = get_number(dialogue[i][1])
                    res[speaker_num]["total"] += 1
                    res[speaker_num][answer_info[0]] += 1
                    answers[speaker_num].append(
                        {
                            "resp": dialogue[i][1],
                            "question": question,
                            "answer": dialogue[i + 1][2],
                            "label": answer_info[0],
                            "model_conf": answer_info[1],
                            "pred_answer": answer_info[2],
                        }
                    )
        with open(
            os.path.join(
                self.temp_path,
                "evasiveness.json",
            ),
            "w",
        ) as filename:
            json.dump([res] + answers, filename)

        return os.path.join(self.temp_path, "evasiveness.json")

from __future__ import annotations

import json
import os
from os import PathLike
from typing import Union, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import OPTICS
from tqdm.auto import tqdm

from expert.core.functional_tools import Rescale
from expert.data.annotation.speech_to_text import (
    get_all_words,
    get_phrases,
    transcribe_video,
)
from expert.data.annotation.summarization import (
    SummarizationEN,
    SummarizationRU,
)
from expert.data.detection.face_detector import FaceDetector
from expert.data.video_reader import VideoReader


class FeatureExtractor:
    """Extracting data from a video presentation.

    The FeatureExtractor extracts information about a speaker from a video presentation.
    Enabled SpeakerDiarization module for diarization of speakers by audio signal, speech
    recognition modules for obtaining stripped phrases of the speaker and the full text
    of the speech, Summarization module for obtaining annotations of expert statements,
    FaceDetector module for detecting and obtaining speaker face embeddings, clustering
    of faces based on the resulting embeddings and extracting images of speakers.

    Returns:
        str: Path to the folder with the results of recognition modules.

    Raises:
        NotImplementedError: If 'language' is not equal to 'en' or 'ru'.
        Warning: If failed to detect faces. Recommended to change 'min_detection_confidence'.

    Example:
        >>> import torch
        >>> extractor = FeatureExtractor(video_path="test_video.mp4", device=torch.device("cuda"))
        >>> extractor.get_features()
        "temp/test_video"
    """

    def __init__(
        self,
        video_path: Union[str, PathLike],
        cache_capacity: Optional[int] = 10,
        device: Optional[Union[torch.device, None]] = None,
        stt_mode: Optional[str] = "local",
        model_selection: Optional[int] = 0,
        min_detection_confidence: Optional[float] = 0.75,
        max_num_faces: Optional[int] = 10,
        max_eps: Optional[int] = 2,
        phrase_duration: Optional[int] = 60,
        lang: Optional[str] = "en",
        get_summary: Optional[bool] = True,
        summary_max_length: Optional[int] = 25,
        summary_percent: Optional[int] = 25,
        sum_sentences_count: Optional[int] = 2,
        sum_max_length: Optional[int] = 300,
        sum_over_chared_postfix: Optional[str] = "...",
        sum_allowed_punctuation: Optional[List] = [
            ",",
            ".",
            "!",
            "?",
            ":",
            "—",
            "-",
            "#",
            "+",
            "(",
            ")",
            "–",
            "%",
            "&",
            "@",
            '"',
            "'",
        ],
        output_dir: Optional[Union[str, PathLike, None]] = None,
        output_img_size: Optional[Union[int, Tuple]] = 512,
        drop_extra: Optional[bool] = True,
    ) -> None:
        """
        Initialization of audio, text and video models parameters.

        Args:
            video_path (Union[str, PathLike]): Path to local video file.
            cache_capacity (Optional[int]): Buffer size for storing frames. Defaults to 10.
            device (Optional[Union[torch.device, None]]): Device type on local machine (GPU recommended). Defaults to None.
            stt_mode (Optional[str]): Model configuration for speech recognition ['server', 'local']. Defaults to 'local'.
            model_selection (Optional[int]): 0 or 1. 0 to select a short-range model that works best
                for faces within 2 meters from the camera, and 1 for a full-range model best for
                faces within 5 meters. Defaults to 0.
            min_detection_confidence (Optional[float]): Minimum confidence value ([0.0, 1.0]) for face
                detection to be considered successful. Defaults to 0.75.
            max_num_faces (Optional[int]): Maximum number of faces to detect. Defaults to 10.
            max_eps (Optional[int]): The maximum distance between two faces. Defaults to 2.
            phrase_duration (Optional[int]): Length of intervals for extracting phrases from speech. Defaults to 60.
            lang (Optional[str]): Speech language for text processing ['ru', 'en']. Defaults to 'en'.
            get_summary (Optional[bool]): Whether or not to annotate the transcribed speech fragments. Defaults to True.
            summary_max_length (Optional[int]): Maximum number of tokens in the generated text. Defaults to 25.
            summary_percent (Optional[int]): Maximum annotation percentage of original text size. Defaults to 25.
            sum_sentences_count (Optional[int]): Sentences count in output annotation. Defaults to 2.
            sum_max_length (Optional[int]): Maximum number of symbols for annotation
                in output annotation. Defaults to 300.
            sum_over_chared_postfix (Optional[str]): End of line character for annotation
                when truncated. Defaults to "...".
            sum_allowed_punctuation (Optional[List]): Allowed punctuation for annotation.
            output_dir (Optional[Union[str, PathLike, None]]): Path to the folder for saving results. Defaults to None.
            output_img_size (Optional[Union[int, Tuple]]): Size of faces extracted from video. Defaults to 512.
            drop_extra (Optional[bool]): Remove intermediate features from the final results. Defaults to True.
        """
        self.video_path = video_path
        self.cache_capacity = cache_capacity
        self.stt_mode = stt_mode
        self.video = VideoReader(
            filename=video_path, cache_capacity=self.cache_capacity
        )

        self._device = torch.device("cpu")
        if device is not None:
            self._device = device

        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence
        self.max_num_faces = max_num_faces
        self.detector = FaceDetector(
            device=self._device,
            model_selection=self.model_selection,
            min_detection_confidence=self.min_detection_confidence,
            max_num_faces=self.max_num_faces,
        )

        if lang not in ["en", "ru"]:
            raise NotImplementedError("'lang' must be 'en' or 'ru'.")
        self.lang = lang

        if output_dir is not None:
            self.temp_path = output_dir
        else:
            basename = os.path.splitext(os.path.basename(video_path))[0]
            self.temp_path = os.path.join("temp", basename)
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        self.stamps = {}
        self.transcribed_text = []
        self.full_text = ""
        self.face_features = []
        self.annotations = []

    @property
    def device(self) -> torch.device:
        """Check the device type.

        Returns:
            torch.device: Device type on local machine.
        """
        return self._device

    def analyze_speech(self) -> Dict:
        """Method for processing audio signal from video."""
        self.transcribed_text = transcribe_video(
            video_path=self.video_path,
            lang=self.lang,
            model=self.stt_mode,
            device=self._device,
        )

        self.stamps = [
            [segment["start"], segment["end"]]
            for segment in self.transcribed_text["segments"]
        ]

        self.transcribed_text, self.full_text = get_all_words(
            transcribation=self.transcribed_text
        )

        with open(
            os.path.join(self.temp_path, "diarization.json"), "w"
        ) as filename:
            json.dump(self.stamps, filename)

        with open(
            os.path.join(self.temp_path, "transcription.json"), "w"
        ) as filename:
            json.dump(self.transcribed_text, filename)
        with open(os.path.join(self.temp_path, "text.txt"), "w") as filename:
            filename.write(self.full_text)

        if self.get_summary:
            if self.lang == "en":
                summarizer = SummarizationEN()
                phrases = get_phrases(
                    self.transcribed_text, duration=self.phrase_duration
                )
                annotations = []
                for phrase in phrases:
                    annotations.append(
                        {
                            "time": phrase["time"],
                            "annotation": summarizer.get_summary(
                                phrase["text"],
                                sentences_count=self.sum_sentences_count,
                                max_length=self.sum_max_length,
                                over_chared_postfix=self.sum_over_chared_postfix,
                                allowed_punctuation=self.sum_allowed_punctuation,
                            ),
                        }
                    )

            if self.lang == "ru":
                summarizer = SummarizationRU()
                phrases = get_phrases(
                    self.transcribed_text, duration=self.phrase_duration
                )
                annotations = []
                for phrase in phrases:
                    annotations.append(
                        {
                            "time": phrase["time"],
                            "annotation": summarizer.get_summary(
                                text=phrase["text"],
                                sentences_count=self.sum_sentences_count,
                                max_length=self.sum_max_length,
                                over_chared_postfix=self.sum_over_chared_postfix,
                                allowed_punctuation=self.sum_allowed_punctuation,
                            ),
                        }
                    )
            self.annotations = annotations
            with open(
                os.path.join(self.temp_path, "summarization.json"), "w"
            ) as filename:
                json.dump(self.annotations, filename)

    def get_features(self) -> str:
        """Method for extracting features from video.

        Returns:
            str: Path to the folder with the results of recognition modules.
        """
        self.analyze_speech()
        fps = self.video.fps

        for start_sec, finish_sec in self.stamps:
            start_frame, finish_frame = int(-(-start_sec * fps // 1)), int(
                -(-finish_sec * fps // 1)
            )
            # Avoiding out-of-range error after diarization.
            finish_frame = (
                finish_frame
                if finish_frame <= len(self.video)
                else len(self.video)
            )

            for frame_idx in range(start_frame, finish_frame, int(fps)):
                face_batch = self.detector.detect(self.video[frame_idx])

                # Recording if face was detected.
                if face_batch is not None:
                    for face_emb, face_location in face_batch:
                        self.face_features.append(
                            {
                                "video_path": self.video_path,
                                "time_sec": int(-(-frame_idx // fps)),
                                "frame_index": frame_idx,
                                "face_bbox": face_location,
                                # Truncate embedding to reduce computational complexity.
                                "face_embedding": face_emb[:150],
                                "speaker_by_audio": "SPEAKER_00",
                                "speaker_by_video": 0,
                            }
                        )

        if len(self.face_features) == 0:
            raise Warning(
                "Failed to detect faces. Try to change 'min_detection_confidence' manually."
            )
        self.face_features = pd.DataFrame(self.face_features)
        self.face_features.to_json(
            os.path.join(self.temp_path, "features.json"), orient="records"
        )

        return self.temp_path
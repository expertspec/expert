from __future__ import annotations

import json
import os
from os import PathLike
from typing import Union, Optional, Dict, List, Tuple

import cv2
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
from expert.data.diarization.speaker_diarization import SpeakerDiarization
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
        ValueError: If 'frame_per_second' is greater than video fps.
        NotImplementedError: If 'language' is not equal to 'en' or 'ru'.
        Warning: If failed to detect faces. Recommended to change 'min_detection_confidence'.
        Warning: If unable to identify unique faces. Recommended to change 'min_cluster_samples'.

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
        frame_per_second: Optional[int] = 3,
        min_cluster_samples: Optional[int] = 25,
        max_eps: Optional[int] = 2,
        sr: Optional[int] = 16000,
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
            frame_per_second (Optional[int]): Frame rate for the face detector. Defaults to 3.
            min_cluster_samples (Optional[int]): Minimum number of samples for clustering. Defaults to 25.
            max_eps (Optional[int]): The maximum distance between two faces. Defaults to 2.
            sr (Optional[int]): Sample rate. Defaults to 16000.
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
        self.frame_step = int(self.video.fps)

        if frame_per_second is not None:
            if frame_per_second > self.video.fps:
                raise ValueError(
                    f"'frame_per_second' must be less than or equal to {self.frame_step}."
                )
            self.frame_step = int(self.video.fps // frame_per_second)

        if lang not in ["en", "ru"]:
            raise NotImplementedError("'lang' must be 'en' or 'ru'.")
        self.lang = lang

        self.min_cluster_samples = min_cluster_samples
        self.max_eps = max_eps
        self.get_summary = get_summary
        self.sr = sr
        self.phrase_duration = phrase_duration
        self.summary_max_length = summary_max_length
        self.summary_percent = summary_percent
        self.sum_sentences_count = sum_sentences_count
        self.sum_max_length = sum_max_length
        self.sum_over_chared_postfix = sum_over_chared_postfix
        self.sum_allowed_punctuation = sum_allowed_punctuation
        self.drop_extra = drop_extra

        if output_dir is not None:
            self.temp_path = output_dir
        else:
            basename = os.path.splitext(os.path.basename(video_path))[0]
            self.temp_path = os.path.join("temp", basename)
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        self.output_img_size = output_img_size
        self.transforms = Rescale(output_img_size)
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
        diarization_pipe = SpeakerDiarization(
            audio=self.video_path, sr=self.sr, device=self._device
        )
        self.stamps = diarization_pipe.apply()
        with open(
            os.path.join(self.temp_path, "diarization.json"), "w"
        ) as filename:
            json.dump(self.stamps, filename)

        self.transcribed_text = transcribe_video(
            video_path=self.video_path,
            lang=self.lang,
            model=self.stt_mode,
            device=self._device,
        )
        self.transcribed_text, self.full_text = get_all_words(
            transcribation=self.transcribed_text
        )

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

        return self.stamps

    def cluster_faces(self) -> pd.DataFrame:
        """Method for clustering the faces of speakers from a video."""
        self.face_features = pd.DataFrame(data=self.face_features)

        # Train a face clusterer and make prediction.
        audio_samples = int(
            self.face_features["speaker_by_audio"].value_counts().mean() // 2
        )
        # If the video is too short, decrease the cluster size.
        min_samples = (
            self.min_cluster_samples
            if audio_samples > self.min_cluster_samples
            else audio_samples
        )
        cl_model = OPTICS(
            metric="euclidean",
            n_jobs=-1,
            cluster_method="xi",
            min_samples=min_samples,
            max_eps=self.max_eps,
        )
        self.face_features["speaker_by_video"] = cl_model.fit_predict(
            np.array(self.face_features["face_embedding"].tolist())
        )

        # Optional step to save memory.
        if self.drop_extra:
            self.face_features = self.face_features.drop(
                columns=["face_embedding"]
            )
            self.face_features = self.face_features.drop_duplicates(
                subset=["time_sec", "speaker_by_video"], keep="first"
            )
            self.face_features = self.face_features.reset_index(drop=True)

        return self.face_features

    def get_features(self) -> str:
        """Method for extracting features from video.

        Returns:
            str: Path to the folder with the results of recognition modules.
        """
        self.stamps = self.analyze_speech()
        fps = self.video.fps

        for speaker in tqdm(self.stamps):
            for start_sec, finish_sec in self.stamps[speaker]:
                start_frame, finish_frame = int(-(-start_sec * fps // 1)), int(
                    -(-finish_sec * fps // 1)
                )
                # Avoiding out-of-range error after diarization.
                finish_frame = (
                    finish_frame
                    if finish_frame <= len(self.video)
                    else len(self.video)
                )
                frames = [
                    frame
                    for frame in range(start_frame, finish_frame)
                    if not frame % self.frame_step
                ]
                for frame_idx in frames:
                    face_batch = self.detector.embed(self.video[frame_idx])

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
                                    "speaker_by_audio": speaker,
                                }
                            )

        if len(self.face_features) == 0:
            raise Warning(
                "Failed to detect faces. Try to change 'min_detection_confidence' manually."
            )
        self.face_features = self.cluster_faces()
        self.face_features.to_json(
            os.path.join(self.temp_path, "features.json"), orient="records"
        )

        return self.temp_path

    def get_face_images(self) -> int:
        """Method for extracting images of speakers faces."""
        images_path = os.path.join(self.temp_path, "faces")
        if not os.path.exists(images_path):
            os.makedirs(images_path)

        idxes = self.face_features[
            self.face_features["speaker_by_video"] != -1
        ]["speaker_by_video"].unique()
        if len(idxes) == 0:
            raise Warning(
                "Unable to identify unique faces. Try to change 'min_cluster_samples' manually."
            )
        for idx, cluster in zip(range(self.max_num_faces), idxes):
            cur_row = self.face_features[
                self.face_features["speaker_by_video"] == cluster
            ].sample()
            cur_frame = self.video[cur_row["frame_index"].item()]
            cur_loc = cur_row["face_bbox"].item()
            cur_face = cur_frame[
                cur_loc[0][1] : cur_loc[0][1] + cur_loc[1][1],
                cur_loc[0][0] : cur_loc[0][0] + cur_loc[1][0],
            ]

            transformed_face = self.transforms(image=cur_face)
            cv2.imwrite(
                os.path.join(images_path, f"{cluster}.jpg"), transformed_face
            )

        return idx

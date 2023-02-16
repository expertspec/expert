from __future__ import annotations

import torch
from sklearn.cluster import OPTICS
from typing import Dict, Tuple, List
from tqdm.auto import tqdm
from os import PathLike
import pandas as pd
import numpy as np
import logging
import json
import cv2
import os

from expert.core.functional_tools import Rescale
from expert.data.video_reader import VideoReader
from expert.data.detection.face_detector import FaceDetector
from expert.data.annotation.speech_to_text import transcribe_video, get_all_words, get_phrases
from expert.data.diarization.speaker_diarization import SpeakerDiarization
from expert.data.annotation.summarization import SummarizationRU, SummarizationEN


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
        NotImplementedError: If 'language' is not equal to 'EN' or 'RU'.
        Warning: If failed to detect faces. Recommended to change 'min_detection_confidence'.
        Warning: If unable to identify unique faces. Recommended to change 'min_cluster_samples'.
    
    Example:
        >>> test_video_path: str = "test_video.mp4"
        >>> extractor = FeatureExtractor(video_path=test_video_path)
        >>> extractor.get_features()
        "temp\\test_video"
    
    Todo:
        * Add Russian speech recognition module.
    """
    def __init__(
        self,
        video_path: str | Pathlike,
        cache_capacity: int = 10,
        device: torch.device | None = None,
        model_selection: int = 0,
        min_detection_confidence: float = 0.75,
        max_num_faces: int = 10,
        frame_per_second: int | None = None,
        min_cluster_samples: int | None = None,
        sr: int = 16000,
        phrase_duration: int = 10,
        language: str = "EN",
        get_summary: bool = True,
        summary_max_length: int = 25,
        summary_percent: int = 25,
        rusum_sentences_count: int = 2,
        rusum_max_length: int = 300,
        rusum_over_chared_postfix: str = "...",
        rusum_allowed_punctuation: List = [",", ".", "!", "?", ":", "—", "-", "#", "+",
                                           "(", ")", "–", "%", "&", "@", '"', "'",],
        output_dir: str | Pathlike | None = None,
        output_img_size: int | Tuple = 512,
        drop_extra: bool = True
    ) -> None:
        """
        Initialization of audio, text and video models parameters.
        
        Args:
            video_path (str | Pathlike): Path to local video file.
            cache_capacity (int, optional): Buffer size for storing frames. Defaults to 10.
            device (torch.device | None, optional): Device type on local machine (GPU recommended). Defaults to None.
            model_selection (int, optional): 0 or 1. 0 to select a short-range model that works best
                for faces within 2 meters from the camera, and 1 for a full-range model best for
                faces within 5 meters. Defaults to 0.
            min_detection_confidence (float, optional): Minimum confidence value ([0.0, 1.0]) for face
                detection to be considered successful. Defaults to 0.75.
            max_num_faces (int, optional): Maximum number of faces to detect. Defaults to 10.
            frame_per_second (int | None, optional): Frame rate for the face detector. Defaults to None.
            min_cluster_samples (int | None, optional): Minimum number of samples for clustering. Defaults to None.
            sr (int, optional): Sample rate. Defaults to 16000.
            phrase_duration (int, optional): Length of intervals for extracting phrases from speech. Defaults to 10.
            get_summary (bool, optional): Whether or not to annotate the transcribed speech fragments. Defaults to True.
            summary_max_length (int, optional): Maximum number of tokens in the generated text. Defaults to 25.
            summary_percent (int, optional): Maximum annotation percentage of original text size. Defaults to 25.
            rusum_sentences_count (int, optional): Sentences count in output annotation
                in Russian. Defaults to 2.
            rusum_max_length (int, optional): Maximum number of symbols for annotation in Russian
                in output annotation. Defaults to 300.
            rusum_over_chared_postfix (str, optional): End of line character for annotation in Russian
                when truncated. Defaults to "...".
            rusum_allowed_punctuation (List, optional): Allowed punctuation for annotation in Russian.
            output_dir (str | Pathlike | None, optional): Path to the folder for saving results
            output_img_size (int | Tuple, optional): Size of faces extracted from video.
            drop_extra (bool, optional): Remove intermediate features from the final results.
        """
        self.video_path = video_path
        self.cache_capacity = cache_capacity
        self.video = VideoReader(filename=video_path, cache_capacity=self.cache_capacity)
        
        self._device = torch.device("cpu")
        if device is not None:
            self._device = device
        
        self.model_selection = model_selection
        self.min_detection_confidence = min_detection_confidence
        self.max_num_faces = max_num_faces
        self.detector = FaceDetector(device=self._device, model_selection=self.model_selection,
            min_detection_confidence=self.min_detection_confidence, max_num_faces=self.max_num_faces)
        self.frame_step = int(self.video.fps)
        
        if frame_per_second is not None:
            if frame_per_second > self.video.fps:
                raise ValueError(f"'frame_per_second' must be less than or equal to {self.frame_step}.")
            self.frame_step = int(self.video.fps // frame_per_second)
        
        if language not in ["EN", "RU"]:
            raise NotImplementedError("'language' must be 'EN' or 'RU'.")
        self.language = language
        
        self.min_cluster_samples = min_cluster_samples
        self.get_summary = get_summary
        self.sr = sr
        self.phrase_duration = phrase_duration
        self.summary_max_length = summary_max_length
        self.summary_percent = summary_percent
        self.rusum_sentences_count = rusum_sentences_count
        self.rusum_max_length = rusum_max_length
        self.rusum_over_chared_postfix = rusum_over_chared_postfix
        self.rusum_allowed_punctuation = rusum_allowed_punctuation
        self.drop_extra= drop_extra
        
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
        diarization_pipe = SpeakerDiarization(audio=self.video_path, sr=self.sr, device=self._device)
        self.stamps = diarization_pipe.apply()
        with open(os.path.join(self.temp_path, "diarization.json"), "w") as filename:
            json.dump(self.stamps, filename)
        
        self.transcribed_text = transcribe_video(video_path=self.video_path, sample_rate=self.sr)
        self.transcribed_text = get_all_words(transcribation=self.transcribed_text)
        with open(os.path.join(self.temp_path, "transcription.json"), "w") as filename:
            json.dump(self.transcribed_text, filename)
        
        self.full_text = " ".join([x["word"] for x in self.transcribed_text])
        with open(os.path.join(self.temp_path, "text.txt"), "w") as filename:
            filename.write(self.full_text)
        
        if self.get_summary:
            if self.language == "EN":
                summarizer = SummarizationEN(device=self._device, max_length=self.summary_max_length,
                    summary_percent=self.summary_percent)
                phrases = get_phrases(self.transcribed_text, duration=self.phrase_duration)
                annotations = []
                for phrase in phrases:
                    annotations.append(
                        {"time": phrase["time"], "annotation": summarizer.get_summary(phrase["text"])}
                    )
            
            if self.language == "RU":
                summarizer = SummarizationRU()
                phrases = get_phrases(self.transcribed_text, duration=self.phrase_duration)
                annotations = []
                for phrase in phrases:
                    annotations.append(
                        {"time": phrase["time"], "annotation": summarizer.get_summary(
                            text=phrase["text"],
                            sentences_count=self.rusum_sentences_count,
                            max_length=self.rusum_max_length,
                            over_chared_postfix=self.rusum_over_chared_postfix,
                            allowed_punctuation=self.rusum_over_chared_postfix
                        )})
            self.annotations = annotations
            with open(os.path.join(self.temp_path, "summarization.json"), "w") as filename:
                json.dump(self.annotations, filename)
        
        return self.stamps
    
    def cluster_faces(self) -> pd.DataFrame:
        """Method for clustering the faces of speakers from a video."""
        self.face_features = pd.DataFrame(data=self.face_features)
        
        # Train a face clusterer and make prediction.
        min_samples = int(self.face_features["speaker_by_audio"].value_counts().mean() // 2)
        cl_model = OPTICS(metric="euclidean", n_jobs=-1, cluster_method="xi", min_samples=min_samples)
        self.face_features["speaker_by_video"] = cl_model.fit_predict(np.array(self.face_features["face_embedding"].tolist()))
        
        # Optional step to save memory.
        if self.drop_extra:
            self.face_features.drop(columns=["face_embedding"], inplace=True)
        
        return self.face_features
    
    def get_features(self) -> str:
        """Method for extracting features from video.
        
        Returns:
            str: Path to the folder with the results of recognition modules.
        """
        self.stamps = self.analyze_speech()
        fps = self.video.fps
        
        logging.warning("Extracting features will take some time. Please, wait.")
        for speaker in tqdm(self.stamps):
            for start_sec, finish_sec in self.stamps[speaker]:
                start_frame, finish_frame = int(-(-start_sec*fps//1)), int(-(-finish_sec*fps//1))
                frames = [frame for frame in range(start_frame, finish_frame) if not frame % self.frame_step]
                for frame_idx in frames:
                    face_batch = self.detector.embed(self.video[frame_idx])
                    
                    # Recording if face was detected.
                    if face_batch is not None:
                        for face_emb, face_location in face_batch:
                            self.face_features.append({
                                "video_path": self.video_path,
                                "time_sec": int(-(-frame_idx//fps)),
                                "frame_index": frame_idx,
                                "face_bbox": face_location,
                                # Truncate embedding to reduce computational complexity.
                                "face_embedding": face_emb[:150],
                                "speaker_by_audio": speaker
                            })
        
        if len(self.face_features) == 0:
            raise Warning("Failed to detect faces. Try to change 'min_detection_confidence' manually.")
        self.face_features = self.cluster_faces()
        self.face_features.to_json(os.path.join(self.temp_path, "features.json"), orient="records")
        n_faces = self.get_face_images() + 1
        
        return self.temp_path
    
    def get_face_images(self) -> int:
        """Method for extracting images of speakers faces."""
        images_path = os.path.join(self.temp_path, "faces")
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        
        idxes = self.face_features[self.face_features["speaker_by_video"] != -1]["speaker_by_video"].unique()
        if len(idxes) == 0:
            raise Warning("Unable to identify unique faces. Try to change 'min_cluster_samples' manually.")
        for idx, cluster in zip(range(self.max_num_faces), idxes):
            cur_row = self.face_features[self.face_features["speaker_by_video"] == cluster].sample()
            cur_frame = self.video[cur_row["frame_index"].item()]
            cur_loc = cur_row["face_bbox"].item()
            cur_face = cur_frame[cur_loc[0][1]:cur_loc[0][1]+cur_loc[1][1],
                                 cur_loc[0][0]:cur_loc[0][0]+cur_loc[1][0]]
            
            transformed_face = self.transforms(image=cur_face)
            cv2.imwrite(os.path.join(images_path, f"{cluster}.jpg"), transformed_face)
        
        return idx
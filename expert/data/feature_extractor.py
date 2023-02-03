from __future__ import annotations

import torch
from sklearn.cluster import OPTICS
from os import PathLike
import pandas as pd
import numpy as np
import json
import cv2
import os

from expert.data.video_reader import VideoReader
from expert.data.detection.face_detector import FaceDetector, Rescale
from expert.data.annotation.speech_to_text import transcribe_video, get_all_words, get_phrases
from expert.data.diarization.speaker_diarization import SpeakerDiarization
from expert.data.annotation.summarization import Summarization


class FeatureExtractor:
    def __init__(
        self,
        video_path: str | Pathlike,
        device: torch.device | None = None,
        max_faces: int = 10,
        frame_per_second: int | None = None,
        get_summary: bool = False,
        summary_step: int = 10,
        output_dir: str | Pathlike | None = None,
        drop_extra: bool = True
    ) -> None:
        """
        Args:
            video_path (str | Pathlike)
            device (torch.device | None, optional)
            max_faces (int, optional)
            frame_per_second (int | None, optional)
            get_summary (bool, optional)
            summary_step (int, optional)
            output_dir (str | Pathlike | None, optional)
            drop_extra (bool, optional)
        """
        super(FeatureExtractor, self).__init__()
        
        self.video_path = video_path
        self.video = VideoReader(filename=video_path)
        self.frame_step = int(self.video.fps)
        if output_dir is not None:
            self.frame_step = frame_per_second
        
        self._device = torch.device("cpu")
        if device is not None:
            self._device = device
        
        self.detector = FaceDetector(device=self._device)
        
        self.max_faces = max_faces
        self.get_summary = get_summary
        self.summary_step = summary_step
        self.drop_extra= drop_extra
        
        if output_dir is not None:
            self.temp_path = output_dir
        else:
            basename = os.path.splitext(os.path.basename(video_path))[0]
            self.temp_path = os.path.join("temp", basename)
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)
        
        self.transforms = Rescale((512, 512))
    
    @property
    def device(self) -> torch.device:
        """Check the device type."""
        return self._device
    
    def analyze_speech(self):
        diarization_pipe = SpeakerDiarization(self.video_path)
        stamps = diarization_pipe.apply()
        
        with open(os.path.join(self.temp_path, "diarization.json"), "w") as filename:
            json.dump(stamps, filename)
        
        transcribed_text = transcribe_video(video_path=self.video_path)
        
        with open(os.path.join(self.temp_path, "transcription.json"), "w") as filename:
            json.dump(transcribed_text, filename)
        
        all_words = get_all_words(transcribation=transcribed_text)
        full_text = " ".join([x["word"] for x in all_words])
        
        with open(os.path.join(self.temp_path, "text.txt"), "w") as filename:
            filename.write(full_text)
        
        if self.get_summary:
            summary = Summarization(device=self._device)
            phrases = get_phrases(all_words, duration=self.summary_step)
            annotations = []
            for phrase in phrases:
                annotations.append(
                    {"time": phrase["time"], "annotation": summary.get_summary(phrase["text"])}
                )
            
            with open(os.path.join(self.temp_path, "summarization.json"), "w") as filename:
                json.dump(annots, filename)
        
        return stamps
    
    def cluster_faces(self):
        data = pd.DataFrame(data=self._features)
        
        # Train a face clusterer and make prediction.
        min_samples = int(data["speaker_by_audio"].value_counts().mean())
        cl_model = OPTICS(metric="euclidean", n_jobs=-1, cluster_method="xi", min_samples=min_samples)
        data["cluster"] = cl_model.fit_predict(np.array(data["face_emb"].tolist()))
        
        # Optional step to save memory.
        if self.drop_extra:
            data.drop(columns=["face_emb"], inplace=True)
        
        return data
    
    def get_features(self):
        stamps = self.analyze_speech()
        fps = self.video.fps
        self._features = []
        
        for speaker in stamps:
            for start_sec, finish_sec in stamps[speaker]:
                start_frame, finish_frame = int(-(-start_sec*fps//1)), int(-(-finish_sec*fps//1))
                frames = [frame for frame in range(start_frame, finish_frame) if not frame % self.frame_step]
                
                for frame_idx in frames:
                    face_batch = self.detector.embed(self.video[frame_idx])
                    
                    # Recording if face was detected.
                    if face_batch is not None:
                        for face_emb, face_location in face_batch:
                            self._features.append({
                                "video_path": self.video_path,
                                "speaker_by_audio": speaker,
                                "time_sec": int(-(-frame_idx//fps)),
                                "frame_idx": frame_idx,
                                "face_bbox": face_location,
                                "face_emb": face_emb[:150],
                            })
        
        self._features = self.cluster_faces()
        self._features.to_json(os.path.join(self.temp_path, "features.json"), orient="records")
        num_faces = self.get_face_images() + 1
        
        return self._features
    
    def get_face_images(self):
        images_path = os.path.join(self.temp_path, "faces")
        if not os.path.exists(images_path):
            os.makedirs(images_path)
        
        idxes = self._features[self._features["cluster"] != -1]["cluster"].unique()
        for idx, cluster in zip(range(self.max_faces), idxes):
            cur_row = self._features[self._features["cluster"] == cluster].sample()
            cur_frame = self.video[cur_row["frame_idx"].item()]
            cur_loc = cur_row["face_bbox"].item()
            cur_face = cur_frame[cur_loc[0][1]:cur_loc[0][1]+cur_loc[1][1],
                                 cur_loc[0][0]:cur_loc[0][0]+cur_loc[1][0]]
            
            transformed_face = self.transforms(image=cur_face)
            cv2.imwrite(os.path.join(images_path, f"{cluster}.jpg"), transformed_face)
        
        return idx
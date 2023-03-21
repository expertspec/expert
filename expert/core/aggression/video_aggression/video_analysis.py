from __future__ import annotations

import os
from os import PathLike
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch

from expert.core.aggression.video_aggression.anglenet import (
    AngleNet,
    classify_rotation,
)
from expert.core.aggression.video_aggression.face_mesh import (
    HEAD_INDEXES,
    LOW_PART,
    UPP_PART,
    FaceMesh,
    preprocess_face_vector,
)
from expert.data.video_reader import VideoReader


class VideoAggression:
    """Extraction of aggression markers by video channel.

    Args:
        video_path (str | PathLike): Path to the local video file.
        features_path (str | PathLike): Path to the result of feature extraction module.
        face_image (str | PathLike): Path to the face image selected by user.
        device (torch.device | None, optional): Device type on local machine (GPU recommended). Defaults to None.
        duration: Length of intervals for extracting features. Defaults to 10.
        static_image_mode (int, optional): Whether to treat the input images as a batch
            of static and possibly unrelated images or a video stream. Defaults to False.
        max_num_faces (int, optional): Maximum number of faces to detect. Defaults to 1.
        refine_landmarks(bool, optional): Whether to further refine the landmark coordinates
            around the eyes, lips and output additional landmarks around the irises. Defaults to True.
        min_detection_confidence(float, optional): Minimum confidence value ([0.0, 1.0]) for face
            detection to be considered successful. Defaults to 0.5.
        min_tracking_confidence (float, optional): Minimum confidence value ([0.0, 1.0]) for the
            face landmarks to be considered tracked successfully. Defaults to 0.5.
    """

    def __init__(
        self,
        video_path: str | PathLike,
        features_path: str | PathLike,
        face_image: str | PathLike,
        device: torch.device | None = None,
        duration: int = 10,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_dir: str | PathLike | None = None,
    ):
        self._device = torch.device("cpu")
        if device is not None:
            self._device = device

        self.duration = duration
        self.video_path = video_path
        self.video = VideoReader(video_path)
        features = pd.read_json(features_path)

        if output_dir is not None:
            self.temp_path = output_dir
        else:
            basename = os.path.splitext(os.path.basename(video_path))[0]
            self.temp_path = os.path.join("temp", basename)
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

        expert_idx = int(os.path.splitext(os.path.basename(face_image))[0])
        self.features = features[features["speaker_by_video"] == expert_idx]
        self.features = (
            self.features.drop_duplicates(subset=["time_sec"])
            .sort_values(by="time_sec")
            .reset_index(drop=True)
        )
        keys = self.features.groupby(by="speaker_by_audio").count()
        self.key = keys.sort_values(
            by="speaker_by_video", ascending=False
        ).index[0]

        self.anglenet = AngleNet(pretrained=True, device=self._device).eval()
        self.mesh_detector = FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @property
    def device(self) -> torch.device:
        """Check the device type.

        Returns:
            torch.device: Device type on local machine.
        """
        return self._device

    def mesh_detection(self) -> List:
        """Determination of face mesh landmarks on video."""
        mesh_landmarks = []
        for row in range(len(self.features)):
            cur_frame = self.video[self.features.loc[row, "frame_index"]]
            cur_loc = self.features.loc[row, "face_bbox"]
            cur_face = cur_frame[
                cur_loc[0][1] : cur_loc[0][1] + cur_loc[1][1],
                cur_loc[0][0] : cur_loc[0][0] + cur_loc[1][0],
            ]
            face_mesh = self.mesh_detector.detect(cur_face, normalize=True)

            if len(face_mesh) != 0:
                head_vector = torch.from_numpy(
                    preprocess_face_vector(face_mesh, head_indexes=HEAD_INDEXES)
                )
                angle_prediction = (
                    self.anglenet(head_vector.to(self._device).float())
                    .detach()
                    .cpu()
                    .numpy()
                )
                mesh_landmarks.append(
                    {
                        "time_sec": self.features.loc[row, "time_sec"],
                        "angles": angle_prediction,
                        "upp_part": np.array(
                            [face_mesh[idx] for idx in LOW_PART]
                        ),
                        "low_part": np.array(
                            [face_mesh[idx] for idx in UPP_PART]
                        ),
                    }
                )

        return mesh_landmarks

    def get_report(self) -> Tuple[List, List, str]:
        """Determination of aggression markers on video."""
        data = pd.DataFrame(data=self.mesh_detection())

        count_upp = 0
        count_low = 0
        count_rap = 0
        count_rot = 0
        state_upp = True
        state_low = True
        state_rap = True
        state_rot = True
        div_vid_agg = []

        # Upper face activity. Calculate difference between the average position of the face and the current.
        mean = data["upp_part"].mean()
        data["upp_coef"] = data["upp_part"].apply(
            lambda x: np.subtract(mean, x).mean()
        )

        quantile_25, quantile_75 = np.quantile(
            data["upp_coef"], 0.25
        ), np.quantile(data["upp_coef"], 0.75)
        quantile_15, quantile_85 = np.quantile(
            data["upp_coef"], 0.15
        ), np.quantile(data["upp_coef"], 0.85)
        data["upp_motion"] = data["upp_coef"].apply(
            lambda x: 1 if x >= quantile_75 or x <= quantile_25 else 0
        )
        data["upp_motion"] = data["upp_coef"].apply(
            lambda x: 2 if x >= quantile_85 or x <= quantile_15 else x
        )

        # Lower face activity. Calculate difference between the average position of the face and the current.
        mean = data["low_part"].mean()
        data["low_coef"] = data["low_part"].apply(
            lambda x: np.subtract(mean, x).mean()
        )

        quantile_25, quantile_75 = np.quantile(
            data["low_coef"], 0.25
        ), np.quantile(data["low_coef"], 0.75)
        quantile_15, quantile_85 = np.quantile(
            data["low_coef"], 0.15
        ), np.quantile(data["low_coef"], 0.85)
        data["low_motion"] = data["low_coef"].apply(
            lambda x: 1 if x >= quantile_75 or x <= quantile_25 else 0
        )
        data["low_motion"] = data["low_coef"].apply(
            lambda x: 2 if x >= quantile_85 or x <= quantile_15 else x
        )

        # Aggregate movements across the face.
        data["sum_motion"] = data[["upp_motion", "low_motion"]].sum(axis=1)

        # Calculate number of head turns.
        data["rotations"] = data["angles"].apply(classify_rotation)

        # Calculate non-verbal markers for time windows.
        start = 0
        for time in range(len(data)):
            if data["time_sec"][time] < data["time_sec"][start] + self.duration:
                finish = time
            else:
                temp_data = data.loc[start:finish, :].reset_index(drop=True)

                for idx in range(len(temp_data)):
                    if temp_data["upp_motion"][idx] > 0:
                        if state_upp:
                            count_upp = count_upp + 1
                            state_upp = False
                    else:
                        state_low = True

                    if temp_data["low_motion"][idx] > 0:
                        if state_low:
                            count_low = count_low + 1
                            state_low = False
                    else:
                        state_low = True

                    if temp_data["sum_motion"][idx] > 2:
                        if state_rap:
                            count_rap = count_rap + 1
                            state_rap = False
                    else:
                        state_rap = True

                    if temp_data["rotations"][idx] > 0:
                        if state_rot:
                            count_rot = count_rot + 1
                            state_rot = False
                    else:
                        state_rot = True

                div_vid_agg.append(
                    {
                        "video_path": self.video_path,
                        "time_sec": float(data["time_sec"][start]),
                        "uppface_activity": float(count_upp),
                        "low_face_activity": float(count_low),
                        "rapid_activity": float(count_rap),
                        "rotate_activity": float(count_rot),
                    }
                )

                start = time
                count_upp = 0
                count_low = 0
                count_rap = 0
                count_rot = 0
                state_upp = True
                state_low = True
                state_rap = True
                state_rot = True

        full_vid_agg = {
            "video_path": self.video_path,
            "uppface_activity": float(
                data[data["upp_motion"] != 0]["upp_motion"].count() / len(data)
            ),
            "low_face_activity": float(
                data[data["low_motion"] != 0]["low_motion"].count() / len(data)
            ),
            "rapid_activity": float(
                data[data["sum_motion"] > 2]["sum_motion"].count() / len(data)
            ),
            "rotate_activity": float(
                data[data["rotations"] != 0]["rotations"].count() / len(data)
            ),
        }

        return (div_vid_agg, full_vid_agg, self.key)

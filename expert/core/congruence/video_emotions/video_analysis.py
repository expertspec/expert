from __future__ import annotations

import os
from os import PathLike
from typing import List, Tuple

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch import nn
from timm.models.layers import conv2d_same
import timm

from expert.core.functional_tools import get_model_weights_url
from expert.data.video_reader import VideoReader


def get_video_emotions(
    video_path: str | PathLike,
    features_path: str | PathLike,
    face_image: str | PathLike,
    device: torch.device | None = None,
    duration: int = 10,
) -> Tuple[List, str]:
    """Classification of expert emotions on video.

    Args:
        video_path (str | PathLike): Path to local video file.
        features_path (str | PathLike): Path to JSON file with information about detected faces.
        face_image (str | PathLike): Path to face image selected by user.
        device (torch.device | None, optional): Device type on local machine (GPU recommended).
            Defaults to None.
    """
    softmax = nn.Softmax(dim=1)
    
    model_name = "enet_b0_8_best_afew.pt"
    url = "https://github.com/HSE-asavchenko/face-emotion-recognition/raw/main/models/affectnet_emotions/enet_b0_8_best_afew.pt"
    cached_dir = get_model_weights_url(model_name=model_name, url=url)
    model = torch.load(cached_dir, map_location=torch.device("cpu"))
    model = model.to(device).eval()
    
    # Declare an augmentation pipeline.
    transforms = A.Compose(
        [
            A.Resize(width=224, height=224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    video = VideoReader(video_path)
    features = pd.read_json(features_path)
    emo_report = []
    emo_sep_report = []

    expert_idx = int(os.path.splitext(os.path.basename(face_image))[0])
    features = features[features["speaker_by_video"] == expert_idx].reset_index(
        drop=True
    )
    key = (
        features.groupby(by="speaker_by_audio")
        .count()
        .sort_values(by="speaker_by_video", ascending=False)
        .index[0]
    )

    for row in range(len(features)):
        current_frame = cv2.cvtColor(
            video[features.loc[row, "frame_index"]], cv2.COLOR_BGR2RGB
        )

        transformed = (
            transforms(image=current_frame)["image"].unsqueeze(0).to(device)
        )
        emotions = model(transformed)
        emotions = softmax(emotions)[0].cpu().detach()
        lim_emotions = softmax(
            torch.Tensor([[emotions[6], emotions[0], emotions[1]]])
        )[0]

        if not lim_emotions.isnan().any().item():
            lim_emotions = lim_emotions.numpy()
            emo_report.append(
                {
                    "time_sec": features.loc[row, "time_sec"],
                    "video_anger": lim_emotions[0],
                    "video_neutral": lim_emotions[1],
                    "video_happiness": lim_emotions[2],
                }
            )
        else:
            emo_report.append(
                {
                    "time_sec": features.loc[row, "time_sec"],
                    "video_anger": 0.0,
                    "video_neutral": 0.0,
                    "video_happiness": 0.0,
                }
            )

    emo_data = pd.DataFrame(data=emo_report)
    emo_data = emo_data.sort_values(by="time_sec").reset_index(drop=True)
    start = 0
    for time in range(len(emo_data)):
        if emo_data["time_sec"][time] < emo_data["time_sec"][start] + duration:
            finish = time
        else:
            temp_data = emo_data.loc[start:finish, :].reset_index(drop=True)

            emo_sep_report.append(
                {
                    "video_path": video_path,
                    "time_sec": float(emo_data["time_sec"][start]),
                    "video_anger": float(temp_data.mean()["video_anger"]),
                    "video_neutral": float(temp_data.mean()["video_neutral"]),
                    "video_happiness": float(
                        temp_data.mean()["video_happiness"]
                    ),
                }
            )

            start = time

    return emo_sep_report, key

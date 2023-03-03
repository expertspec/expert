import torch
from torch import nn
import torchvision.transforms as torchtransforms
from typing import Tuple, List
from os import PathLike
import pandas as pd
import cv2
import os

from app.libs.expert.expert.data.detection.face_detector import Rescale, Normalize, ToTensor
from app.libs.expert.expert.data.video_reader import VideoReader
from app.libs.expert.expert.core.congruence.video_emotions.video_model import DAN


def get_video_emotions(
    video_path: str | PathLike,
    features_path: str | PathLike,
    image: str | PathLike,
    device: torch.device | None = None
) -> Tuple[List, str]:
    """Classification of expert emotions on video.
    
    Args:
        path (str): Path to video.
        video_path (str): Path to JSON file with information about detected faces.
        image (str): Path to image selected by user.
    """
    softmax = nn.Softmax(dim=1)
    att_resnet = DAN(pretrained=True, device=device).eval()
    transforms = torchtransforms.Compose([
        Rescale((224, 224)),
        ToTensor(),
        Normalize()
    ])
    
    video = VideoReader(video_path)
    features = pd.read_json(features_path)
    emo_report = []
    emo_sep_report = []
    
    expert_idx = int(os.path.splitext(os.path.basename(image))[0])
    features = features[features["cluster"] == expert_idx].reset_index(drop=True)
    key = features.groupby(by="speaker_by_audio").count().sort_values(by="cluster", ascending=False).index[0]
    
    for row in range(len(features)):
        current_frame = cv2.cvtColor(video[features.loc[row, "frame_idx"]], cv2.COLOR_BGR2RGB)
        current_loc = features.loc[row, "face_bbox"]
        current_face = current_frame[current_loc[0][1]:current_loc[0][1]+current_loc[1][1],
                                     current_loc[0][0]:current_loc[0][0]+current_loc[1][0]]
        
        transformed = transforms(current_frame).unsqueeze(0).to(device)
        emotions, _, _ = att_resnet(transformed)
        emotions = softmax(emotions)[0].cpu().detach()
        lim_emotions = softmax(torch.Tensor([[emotions[6], emotions[0], emotions[1]]]))[0]
        
        if not lim_emotions.isnan().any().item():
            lim_emotions = lim_emotions.numpy()
            emo_report.append({
                "time_sec": features.loc[row, "time_sec"],
                "video_anger": lim_emotions[0],
                "video_neutral": lim_emotions[1],
                "video_happiness": lim_emotions[2]
            })
        else:
            emo_report.append({
                "time_sec": features.loc[row, "time_sec"],
                "video_anger": 0.0,
                "video_neutral": 0.0,
                "video_happiness": 0.0
            })
    
    emo_data = pd.DataFrame(data=emo_report)
    emo_data = emo_data.sort_values(by="time_sec").reset_index(drop=True)
    start = 0
    for time in range(len(emo_data)):
        if emo_data["time_sec"][time] < emo_data["time_sec"][start]+10:
            finish = time
        else:
            temp_data = emo_data.loc[start:finish, :].reset_index(drop=True)
            
            emo_sep_report.append({
                "video_path": video_path,
                "time_sec": float(emo_data["time_sec"][start]),
                "video_anger": float(temp_data.mean()["video_anger"]),
                "video_neutral": float(temp_data.mean()["video_neutral"]),
                "video_happiness": float(temp_data.mean()["video_happiness"]),
            })
            
            start = time
    
    return emo_sep_report, key
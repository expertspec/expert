import json
import torch
import pandas as pd
from typing import Dict
from os import PathLike
import os

from app.libs.expert.expert.core.congruence.video_emotions.video_analysis import get_video_emotions
from app.libs.expert.expert.core.congruence.text_emotions.text_analysis import get_text_emotions
from app.libs.expert.expert.core.congruence.audio_emotions.audio_analysis import AudioAnalysis


def align_timestamps(time_sec):
        return time_sec - time_sec % 10


class CongruenceDetection:
    """Determination of expert emotions congruence.
    
    Args:
        video_path (str): Path to original video.
        features_path (str): Path to JSON file with information about detected faces.
        face_image_path (str): Path to image selected by user.
        words_path (str): Path to JSON file with text transcription.
        diarization_path (str): Path to JSON file with diarization information.
        save_to (str): Path to save JSON with information emotions and congruence.
    """
    def __init__(
        self,
        video_path: str | PathLike,
        features_path: str | PathLike,
        face_image_path: str | PathLike,
        words_path: str | PathLike,
        diarization_path: str | PathLike,
        stamps_path: Dict | None = None,
        device: torch.device | None = None,
        output_dir: str | PathLike | None = None,
    ):
        self.video_path = video_path
        self.features_path = features_path
        self.stamps_path = stamps_path
        self.face_image_path = face_image_path
        self.words_path = words_path
        self.diarization_path = diarization_path
        
        self._device = torch.device("cpu")
        if device is not None:
            self._device = device
        
        with open(self.diarization_path, "r") as file:
            self.stamps = json.load(file)
        
        if output_dir is not None:
            self.temp_path = output_dir
        else:
            basename = os.path.splitext(os.path.basename(video_path))[0]
            self.temp_path = os.path.join("temp", basename)
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)
    
    @property
    def device(self) -> torch.device:
        """Check the device type."""
        return self._device
    
    def get_video_state(self):
        video_data, key = get_video_emotions(
            video_path=self.video_path,
            features_path=self.features_path,
            image=self.face_image_path,
            device=self._device
        )
        video_data = pd.DataFrame(data=video_data)
        video_data["time_sec"] = video_data["time_sec"].apply(align_timestamps)
        
        video_data = video_data.drop_duplicates(subset=["time_sec"]).sort_values(by="time_sec").reset_index(drop=True)
        
        return video_data, key
    
    def get_audio_state(self, key):
        audio_model = AudioAnalysis(
            video_path=self.video_path,
            stamps=self.stamps,
            speaker=key,
            device=self._device)
        
        audio_data = audio_model.predict()
        audio_data = pd.DataFrame(data=audio_data)
        audio_data["time_sec"] = audio_data["time_sec"].apply(align_timestamps)
        
        audio_data = audio_data.drop_duplicates(subset=["time_sec"]).sort_values(by="time_sec").reset_index(drop=True)
        
        return audio_data
    
    def get_text_state(self, key):
        text_data = get_text_emotions(words_path=self.words_path, stamps=self.stamps, key=key, device=self._device)
        text_data = pd.DataFrame(data=text_data)
        text_data["time_sec"] = text_data["time_sec"].apply(align_timestamps)
        
        text_data = text_data.drop_duplicates(subset=["time_sec"]).sort_values(by="time_sec").reset_index(drop=True)
        
        return text_data
    
    def get_congruence(self):
        video_data, key = self.get_video_state()
        audio_data = self.get_audio_state(key=key)
        text_data = self.get_text_state(key=key)
        
        cong_data = video_data.join(audio_data.set_index("time_sec"), how="inner", on="time_sec")
        cong_data = cong_data.join(text_data.set_index("time_sec"), how="inner", on="time_sec")
        cong_data = cong_data.sort_values(by="time_sec").reset_index(drop=True)
        
        neutral_std = cong_data.loc[:, ["video_neutral", "audio_neutral", "text_neutral"]].std(axis=1)
        anger_std = cong_data.loc[:, ["video_anger", "audio_anger", "text_anger"]].std(axis=1)
        happiness_std = cong_data.loc[:, ["video_happiness", "audio_happiness", "text_happiness"]].std(axis=1)
        cong_data["congruence"] = (neutral_std + anger_std + happiness_std) / 1.5
        
        # Get and save data with all emotions.
        emotions_data = dict()
        emotions_data["video"] = video_data.to_dict(orient="records")
        emotions_data["audio"] = audio_data.to_dict(orient="records")
        emotions_data["text"] = text_data.to_dict(orient="records")
        
        with open(os.path.join(self.temp_path, "emotions.json"), 'w') as filename:
            json.dump(emotions_data, filename)
        
        cong_data[["video_path", "time_sec", "congruence"]].to_json(
            os.path.join(self.temp_path, "congruence.json"), orient="records"
        )
        
        return cong_data
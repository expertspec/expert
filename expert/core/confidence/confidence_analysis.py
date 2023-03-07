from __future__ import annotations

import json
import os
import ffmpeg

import torch
import pandas as pd

from expert.core.confidence.liedet.data import VideoReader
from expert.core.confidence.liedet.models.e2e import LieDetectorRunner
from expert.core.confidence.liedet.models.registry import build

from mmcv.utils import Config


class ConfidenceDetector:
    def __init__(
        self,
        path_to_video: str,
        path_to_report: str,
        path_to_image: str,
        path_to_diarization: str,
        device: torch.device | None = None,
        save_to: str = "app\\temp"
    ) -> None:
        """Determination of expert's confidence

        Args:
            path_to_video (str): Path to original video.
            path_to_report (str): Path to JSON file with information about detected faces.
            path_to_image (str): Path to image selected by user.
            path_to_diarization (str): Path to JSON file with diarization information.
            save_to (str): Path to save JSON with information emotions and congruence.

        Returns:
            str: Path to JSON file with information about confidence
        """
        self._device = torch.device("cpu")
        if device is not None:
            self._device = device
            
        self.path_to_video = path_to_video
        self.path_to_report = path_to_report
        self.path_to_image = path_to_image
        self.path_to_diarization = path_to_diarization
        self.save_to = save_to
        
    def convert(self, target_fps=30):
        # Get fps info
        probe = ffmpeg.probe(self.path_to_video)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        orig_fps = int(video_info['r_frame_rate'].split('/')[0])
        if orig_fps != target_fps:
            try:
                input = ffmpeg.input(self.path_to_video)
                video = input.video
                audio = input.audio
                video = ffmpeg.filter(video, 'fps', fps=target_fps, round='up')
                converted_name = os.path.basename(self.path_to_video).split('.')
                converted_name = converted_name[0] + '_conv' + '.' + converted_name[1]
                out = ffmpeg.output(video, audio, os.path.join(
                    os.path.dirname(self.path_to_video), converted_name
                    ))
                ffmpeg.run(out, capture_stderr=True)
                
                return f"upload_file/{converted_name}"
            except ffmpeg.Error as conv_error:
                print('stdout:', conv_error.stdout.decode('utf8'))
                print('stderr:', conv_error.stderr.decode('utf8'))
                raise conv_error
        else:
            print("Video has appropriate fps")
            return self.path_to_video
        
    def get_confidence(self):
        cfg = "./expert/core/confidence/configs/landmarks_audio_transformer.py"
        video_path = self.path_to_video

        face_report = pd.read_json(self.path_to_report)
        expert_idx = int(os.path.splitext(os.path.basename(self.path_to_image))[0])
        face_report = face_report[face_report["cluster"] == expert_idx].reset_index(
            drop=True
        )
        key = (
            face_report.groupby(by="speaker_by_audio")
            .count()
            .sort_values(by="cluster", ascending=False)
            .index[0]
        )

        with open(self.path_to_diarization, "r") as f:
            stamps = json.load(f)

        stamps = stamps[key]

        cfg = Config.fromfile(cfg)

        vr = VideoReader(uri=video_path, **cfg.dataset)
        length = len(vr)

        model = build(cfg.model)
        model = model.to(self._device)
        runner = LieDetectorRunner(model=model)

        report = []
        if stamps:
            for stamp in stamps:
                current_time = stamp[0]  # Initial time
                for start in range(
                    stamp[0] * 3 * cfg.video_fps, stamp[1] * 3 * cfg.video_fps, 3 * cfg.window
                ):
                    try:
                        sample = vr[start : start + 3 * cfg.window]
                    except:
                        sample = vr[start:]
                    # To fix overdrawing bag
                    if sample["video_frames"].size()[0] != cfg.window:
                        sample["video_frames"] = sample["video_frames"][: cfg.window].to(self._device)
                        temporary = sample["audio_frames"][0][
                            : cfg["window_secs"] * cfg["audio_fps"]
                        ]
                        temporary.unsqueeze_(dim=0)
                        sample["audio_frames"] = temporary.to('cuda')
                    sample['video_frames'] = sample["video_frames"].to(self._device)
                    sample["audio_frames"] = sample["audio_frames"].to(self._device)
                    predict = runner.predict_sample(sample)
                    if isinstance(predict, str):
                        report.append({"time_sec": current_time, "confidence": predict})
                    else:
                        report.append(
                            {
                                "time_sec": current_time,
                                "confidence": float(predict.to("cpu").detach().numpy()[0][0]),
                            }
                        )  
                    current_time += cfg.window / cfg.video_fps

        # If information about timestamps is empty
        else:
            for start in range(0, length, 3 * cfg.window):
                try:
                    sample = vr[start : start + 3 * cfg.window]
                except:
                    sample = vr[start:]
                if sample["video_frames"].size()[0] != cfg.window:
                    sample["video_frames"] = sample["video_frames"][: cfg.window].to(self._device)
                    temporary = sample["audio_frames"][0][
                        : cfg["window_secs"] * cfg["audio_fps"]
                    ].to(self._device)
                    temporary.unsqueeze_(dim=0)
                    sample["audio_frames"] = temporary
                sample['video_frames'] = sample["video_frames"].to(self._device)
                sample["audio_frames"] = sample["audio_frames"].to(self._device)
                predict = runner.predict_sample(sample)
                current_time = 0
                if isinstance(predict, str):
                    report.append({"time_sec": current_time, "confidence": predict})
                else:
                    report.append(
                        {
                            "time_sec": current_time,
                            "confidence": float(predict.to("cpu").detach().numpy()[0][0]),
                        }
                    )  
                current_time += cfg.window / cfg.video_fps

        # Save results to temporary path.
        temp_path = os.path.splitext(os.path.basename(self.path_to_video))[0]
        if not os.path.exists(os.path.join(self.save_to, temp_path)):
            os.makedirs(os.path.join(self.save_to, temp_path))

        with open(os.path.join(self.save_to, temp_path, "confidence.json"), "w") as filename:
            json.dump(report, filename)

        return os.path.join(self.save_to, temp_path, "confidence.json")


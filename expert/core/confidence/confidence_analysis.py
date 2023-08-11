from __future__ import annotations

import json
import os
from os import PathLike

import ffmpeg
import pandas as pd
import torch
from mmcv.utils import Config

from expert.core.confidence.liedet.data import VideoReader
from expert.core.confidence.liedet.models.e2e import LieDetectorRunner
from expert.core.confidence.liedet.models.registry import build


class ConfidenceDetector:
    """Determination of expert confidence.

    Args:
        video_path (str | PathLike): Path to original video.
        features_path (str | PathLike): Path to the result of feature extraction module.
        face_image (str | PathLike): Path to the face image selected by user.
        diarization_path (str | PathLike): Path to the result of diarization module.
        output_dir (str | Pathlike | None, optional): Path to the folder for saving results. Defaults to None.
    
    Returns:
        str: Path to the contradiction report.

    Raises:
        ffmpeg.Error if problems with convertation are detected
    """
    def __init__(
        self,
        video_path: str | PathLike,
        features_path: str | PathLike ,
        face_image: str | PathLike,
        diarization_path: str | PathLike,
        device: torch.device | None = None,
        output_dir: str | PathLike | None = None,
    ) -> None:

        self._device = torch.device("cpu")
        if device is not None:
            self._device = device

        self.video_path = video_path
        self.features_path = features_path
        self.face_image = face_image
        self.diarization_path = diarization_path

        if output_dir is not None:
            self.temp_path = output_dir
        else:
            basename = os.path.splitext(os.path.basename(video_path))[0]
            self.temp_path = os.path.join("temp", basename)
        if not os.path.exists(self.temp_path):
            os.makedirs(self.temp_path)

    def convert(self, target_fps=30):
        # Get fps info.
        probe = ffmpeg.probe(self.video_path)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        orig_fps = int(video_info["r_frame_rate"].split("/")[0])
        if orig_fps != target_fps:
            try:
                input = ffmpeg.input(self.video_path)
                video = input.video
                audio = input.audio
                video = ffmpeg.filter(video, "fps", fps=target_fps, round="up")
                converted_name = os.path.basename(self.video_path).split(".")
                converted_name = converted_name[0] + "_conv" + "." + converted_name[1]
                out = ffmpeg.output(
                    video,
                    audio,
                    os.path.join(os.path.dirname(self.video_path), converted_name),
                )
                ffmpeg.run(out, capture_stderr=True)

                return f"upload_file/{converted_name}"
            except ffmpeg.Error as conv_error:
                print("stdout:", conv_error.stdout.decode("utf8"))
                print("stderr:", conv_error.stderr.decode("utf8"))
                raise conv_error
        else:
            print("Video has appropriate fps")
            return self.video_path

    def get_confidence(self):
        cfg = "./expert/core/confidence/configs/landmarks_audio_transformer.py"
        video_path = self.video_path

        face_report = pd.read_json(self.features_path)
        expert_idx = int(os.path.splitext(os.path.basename(self.face_image))[0])
        face_report = face_report[
            face_report["speaker_by_video"] == expert_idx
        ].reset_index(drop=True)
        key = (
            face_report.groupby(by="speaker_by_audio")
            .count()
            .sort_values(by="speaker_by_video", ascending=False)
            .index[0]
        )

        with open(self.diarization_path, "r") as f:
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
                    int(stamp[0]) * 3 * cfg.video_fps,
                    int(stamp[1]) * 3 * cfg.video_fps,
                    3 * cfg.window,
                ):
                    try:
                        sample = vr[start : start + 3 * cfg.window]
                    except Exception:
                        sample = vr[start:]
                    # To fix overdrawing bag
                    if sample["video_frames"].size()[0] != cfg.window:
                        sample["video_frames"] = sample["video_frames"][
                            : cfg.window
                        ].to(self._device)
                        temporary = sample["audio_frames"][0][
                            : cfg["window_secs"] * cfg["audio_fps"]
                        ]
                        temporary.unsqueeze_(dim=0)
                        sample["audio_frames"] = temporary.to("cuda")
                    sample["video_frames"] = sample["video_frames"].to(self._device)
                    sample["audio_frames"] = sample["audio_frames"].to(self._device)
                    predict = runner.predict_sample(sample)
                    if isinstance(predict, str):
                        report.append({"time_sec": current_time, "confidence": predict})
                    else:
                        report.append(
                            {
                                "time_sec": current_time,
                                "confidence": float(
                                    predict.to("cpu").detach().numpy()[0][0]
                                ),
                            }
                        )
                    current_time += cfg.window / cfg.video_fps

        # If information about timestamps is empty.
        else:
            for start in range(0, length, 3 * cfg.window):
                try:
                    sample = vr[start : start + 3 * cfg.window]
                except Exception:
                    sample = vr[start:]
                if sample["video_frames"].size()[0] != cfg.window:
                    sample["video_frames"] = sample["video_frames"][: cfg.window].to(
                        self._device
                    )
                    temporary = sample["audio_frames"][0][
                        : cfg["window_secs"] * cfg["audio_fps"]
                    ].to(self._device)
                    temporary.unsqueeze_(dim=0)
                    sample["audio_frames"] = temporary
                sample["video_frames"] = sample["video_frames"].to(self._device)
                sample["audio_frames"] = sample["audio_frames"].to(self._device)
                predict = runner.predict_sample(sample)
                current_time = 0
                if isinstance(predict, str):
                    report.append({"time_sec": current_time, "confidence": predict})
                else:
                    report.append(
                        {
                            "time_sec": current_time,
                            "confidence": float(
                                predict.to("cpu").detach().numpy()[0][0]
                            ),
                        }
                    )
                current_time += cfg.window / cfg.video_fps

        # Save results to temporary path.
        with open(os.path.join(self.temp_path, "confidence.json"), "w") as filename:
            json.dump(report, filename)

        return os.path.join(self.temp_path, "confidence.json")

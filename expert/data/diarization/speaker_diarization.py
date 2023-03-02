from __future__ import annotations

import torch
from torch import Tensor
from pyannote.audio import Pipeline
from decord import AudioReader, bridge
from typing import Dict
from os import PathLike

from expert.data.diarization.audio_transforms import (
    rttm_to_timestamps, json_to_timestamps, merge_intervals,
    separate_marks_for_speakers, get_rounded_intervals
)

bridge.set_bridge(new_bridge="torch")


class SpeakerDiarization:
    """Speaker diarization module by audio signal."""

    def __init__(
        self,
        audio: PathLike | Tensor,
        sr: int = 16000,
        rttm_file: str | PathLike | None = None,
        device: torch.device | None = None
    ) -> None:
        """
        Args:
            audio (PathLike | Tensor): Path to file with audio channel or Tensor.
            sr (int, optional): Sample rate. Defaults to 16000.
            rttm_file (str | PathLike | None, optional): Path to annotation file. Defaults to None.
            device (torch.device | None, optional): Device type on local machine (GPU recommended). Defaults to None.

        Raises:
            TypeError: If audio isn't in appropriate form.
        """

        if isinstance(audio, Tensor):
            self.audio = audio
        elif isinstance(audio, str):
            self.path = audio
            self.audio = AudioReader(audio, sample_rate=sr, mono=True)
            self.audio = self.audio[:]
        else:
            raise TypeError("Audio isn't in appropriate form.")

        self.sr = sr
        self.rttm_file = rttm_file
        self._device = torch.device("cpu")
        if device is not None:
            self._device = device

        token = "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"
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1", use_auth_token=token
        )

    @property
    def device(self) -> torch.device:
        """Check the device type.

        Returns:
            torch.device: Device type on local machine.
        """
        return self._device

    def apply(self) -> Dict:
        """Extract timestamps for every unique speaker. Will create rttm file if 'rttm_file' is set.

        Returns:
            Dict: Dictionary {Speaker ID: List with timestamps}.
        """
        self.annotation = self.pipeline(
            {"waveform": self.audio, "sample_rate": self.sr}
        )

        if self.rttm_file:
            with open(self.rttm_file, "w") as file:
                self.annotation.write_rttm(file)
            self.stamps = rttm_to_timestamps(self.rttm_file)
        else:
            self.stamps = self.annotation.for_json()
            self.stamps = json_to_timestamps(self.stamps)

        self.stamps = separate_marks_for_speakers(self.stamps)
        self.stamps = get_rounded_intervals(self.stamps)
        self.stamps = merge_intervals(self.stamps)

        return self.stamps

from __future__ import annotations

from os import PathLike
from typing import Dict

import numpy as np
import torch
import torchaudio
from decord import AudioReader, bridge
from torch import nn

from expert.core.congruence.audio_emotions.audio_model import AudioModel


bridge.set_bridge(new_bridge="torch")


class AudioAnalysis:
    """Classification of emotions on audio."""

    def __init__(
        self,
        video_path: str | PathLike,
        stamps: Dict,
        speaker: str = "SPEAKER_00",
        sr: int = 44100,
        duration: int = 10,
        device: torch.device | None = None,
    ):
        """
        Args:
            video_path (str | PathLike): Path to local video file.
            stamps (Dict, optional): Dictionary with information about detected speakers.
            speaker (str, optional): Expert selected by user.
                Defaults to 'SPEAKER_00'.
            sr (int, optional): Sample rate. Defaults to 44100.
            duration (int, optional): Length of intervals for extracting features.
                Defaults to 10.
            device (torch.device | None, optional): Device type on local machine (GPU recommended).
                Defaults to None.
        """
        self.sr = sr
        if isinstance(video_path, (str, PathLike)):
            self.path = video_path
            self.audio = AudioReader(
                video_path, sample_rate=self.sr, mono=True
            )[:]
        else:
            raise TypeError

        self._device = torch.device("cpu")
        if device is not None:
            self._device = device

        self.duration = duration
        self.stamps = stamps
        self.speaker = speaker
        self.model = AudioModel(device=self._device)

        self.num_samples = 3 * sr
        self.target_sample_rate = sr

        self.predicts = []

    @property
    def device(self) -> torch.device:
        """Check the device type.

        Returns:
            torch.device: Device type on local machine.
        """
        return self._device

    def predict(self):
        """Create report with information of the key emotions."""
        softmax = nn.Softmax(dim=1)
        if self.stamps[self.speaker]:
            for stamp in self.stamps[self.speaker]:
                current_time = stamp[0]
                fragment = self.audio[0][
                    stamp[0] * self.sr : stamp[1] * self.sr
                ]
                self.chunks = self._chunkizer(
                    self.duration, fragment.numpy(), self.sr
                )

                for num, chunk in enumerate(self.chunks):
                    parts_predict = []
                    self.chunk_parts = self._chunkizer(3, chunk, self.sr)
                    self.chunk_parts = [
                        torch.Tensor(i) for i in self.chunk_parts
                    ]
                    self.test = []
                    for i in range(len(self.chunk_parts)):
                        w = self.chunk_parts[i]
                        w.unsqueeze_(0)
                        w = self._cut_if_necessary(w)
                        w = self._right_pad_if_necessary(w)

                        mfcc = torchaudio.transforms.MFCC(
                            sample_rate=self.sr, n_mfcc=13
                        )(w)

                        mfcc = np.transpose(mfcc.numpy(), (1, 2, 0))
                        mfcc = np.transpose(mfcc, (2, 0, 1)).astype(np.float32)

                        self.test.append(
                            torch.tensor(mfcc, dtype=torch.float).to(
                                self._device
                            )
                        )

                    self.model.eval()
                    for i in range(len(self.test)):
                        c = self.test[i]
                        c.unsqueeze_(0)

                        logits = self.model(c)[0].cpu().detach()
                        lim_emotions = softmax(
                            torch.Tensor([[logits[0], logits[3], logits[2]]])
                        )[0].numpy()
                        parts_predict.append(lim_emotions)
                    parts_predict = np.array(parts_predict)
                    self.predicts.append(
                        {
                            "time_sec": float(current_time),
                            "audio_anger": float(parts_predict[:, [0]].mean()),
                            "audio_neutral": float(
                                parts_predict[:, [1]].mean()
                            ),
                            "audio_happiness": float(
                                parts_predict[:, [2]].mean()
                            ),
                        }
                    )
                    current_time += len(chunk) // self.sr
        else:
            raise "No stamps."

        return self.predicts

    # Audio processing functions.
    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]

        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)

        return signal

    def _chunkizer(self, chunk_length, audio, sr):
        duration = audio.shape[0] / sr
        num_chunks = int(-(-duration // chunk_length))
        chunks = []
        for i in range(num_chunks):
            chunks.append(
                audio[i * chunk_length * sr : (i + 1) * chunk_length * sr]
            )

        return chunks

import os
import uuid

import torch
import torch.nn as nn
import torchaudio
from torch import Tensor

from ...registry import registry
from .main import main


@registry.register_module()
class AudioFeatures(nn.Module):
    """Audio Features extractor."""

    def __init__(self, fps: int = 30, chunk_length=1, sr=48000, normalization: bool = True):
        """
        Args:
            fps (int, optional): number of video frames per second.
                Defaults to 30.
            chunk_length (int, optional): number of second per chunk.
                Defaults to 1.
            sr (int, optional): audio sample rate.
                Defaults to 48000.
            normalization (bool, optional): boolean flag to normalize audio features.
                Defaults to True.
        """
        super().__init__()

        self.video_fps = fps
        self.chunk_length = chunk_length
        self.sample_rate = sr
        self.normalization = normalization

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Forwards audio frames to features extractor.

        Args:
            x (Tensor): batch of input audio frames.

        Returns:
            Tensor: batch of extracted audio features.
        """
        batch_size = x.size(0)
        device = x.device

        x = x.cpu()
        h = []
        for i in range(batch_size):
            tmp_path = uuid.uuid4()
            tmp_path = f"/tmp/{tmp_path}.wav"
            torchaudio.save(tmp_path, x[i], self.sample_rate)

            hi = main(
                audio_path=tmp_path,
                fps=self.video_fps,
                normalization=self.normalization,
                sr=self.sample_rate,
                chunk_length=self.chunk_length,
                csv=False,
            )
            h.append(hi)

            os.remove(tmp_path)

        return torch.stack(h, dim=0).to(device)

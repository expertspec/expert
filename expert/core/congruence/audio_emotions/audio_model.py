from __future__ import annotations

import torch
from torch import nn
import gdown
import os


from app.libs.expert.expert.core.utils import get_torch_home



class AudioModel(nn.Module):
    """Model for emotion classification by audio signal."""

    def __init__(
        self,
        pretrained: bool = True,
        device: torch.device | None = None
    ) -> None:
        """
        Args:
            pretrained (bool, optional): Whether or not to load saved pretrained weights. Defaults to True.
            device (torch.device | None, optional): Device type on local machine (GPU recommended). Defaults to None.
        """
        super(AudioModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(512)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=512,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(256)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(128)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.BatchNorm2d(64)
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.3)
        self.linear = nn.Linear(5504, 7)
        self.softmax = nn.Softmax(dim=1)
        self._device = torch.device("cpu")

        if pretrained:
            url = "https://drive.google.com/uc?export=view&id=1DU5pu9D0BSvXBCj_J3kqzgdallxebcLX"
            model_name = "audio_model.pth"

            cached_file = get_model_weights(model_name=model_name, url=url)
            state_dict = torch.load(cached_file, map_location=self._device)
            self.load_state_dict(state_dict, strict=True)

        if device is not None:
            self._device = device
            self.to(self._device)

    @property
    def device(self) -> torch.device:
        """Check the device type.

        Returns:
            torch.device: Device type on local machine.
        """
        return self._device

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.dropout(self.flatten(x))

        logits = self.linear(x)
        predictions = self.softmax(logits)

        return predictions

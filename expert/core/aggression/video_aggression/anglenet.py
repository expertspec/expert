from __future__ import annotations

import os
import torch
from torch import nn, Tensor
import numpy as np

from app.libs.expert.expert.core.functional_tools import get_model_weights


class AngleNet(nn.Module):
    """AngleNet implementation.

    Model implementation for head rotation angles prediction using face mesh.

    Example:
        >>> import torch
        >>> anglenet = AngleNet(pretrained=True, device=torch.device('cuda')).eval()
    """

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

        super(AngleNet, self).__init__()

        self.layer_1 = nn.Sequential(
            nn.Linear(in_features=128*3, out_features=512, bias=True),
            nn.Dropout(p=0.3),
            nn.LeakyReLU()
        )

        self.layer_2 = nn.Sequential(
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.Dropout(p=0.2),
            nn.LeakyReLU()
        )

        self.layer_3 = nn.Sequential(
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.Dropout(p=0.2),
            nn.LeakyReLU()
        )

        self.layer_4 = nn.Sequential(
            nn.Linear(in_features=128, out_features=64, bias=True),
            nn.Dropout(p=0.1),
            nn.LeakyReLU()
        )

        self.layer_5 = nn.Sequential(
            nn.Linear(in_features=64, out_features=32, bias=True),
            nn.Dropout(p=0.1),
            nn.LeakyReLU()
        )

        self.fc = nn.Linear(in_features=32, out_features=3)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(tensor=m.weight, mean=0, std=0.01)
                nn.init.constant_(tensor=m.bias, val=0)

        self._device = torch.device("cpu")
        if pretrained:
            model_name = "anglenet.pth"
            url = "https://drive.google.com/uc?export=view&id=15cUOL9u_Gva7nYdqM9UgOleBS6xboh-r"
            cached_file = get_model_weights(model_name=model_name, url=url)
            state_dict = torch.load(cached_file, map_location=self._device)
            self.load_state_dict(state_dict, strict=True)

        if device is not None:
            self._device = device
            self.to(device)

    @property
    def device(self) -> torch.device:
        """Check the device type.

        Returns:
            torch.device: Device type on local machine.
        """
        return self._device

    def forward(self, x: Tensor) -> Tensor:
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.fc(x)

        return x


def classify_rotation(angle_predictions: np.ndarray) -> int:
    """Classification of turning away by the angles of head rotation.

    Args:
        angle_predictions (np.ndarray): Head angle predictions represented as numpy ndarray.
    """

    # Comparing angle values to the threshold value in radians.
    rotation_threshold = np.radians(25)
    if np.absolute(angle_predictions[0]) >= rotation_threshold or np.absolute(angle_predictions[1]) >= rotation_threshold:
        return 1
    return 0

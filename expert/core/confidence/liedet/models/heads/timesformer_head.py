from __future__ import annotations

import torch.nn as nn
from mmcv.cnn import trunc_normal_init
from torch import Tensor


class TimeSformerHead(nn.Module):
    """Classification head for TimeSformer."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 768,
        init_std: float = 0.02,
        **kwargs,
    ) -> None:
        """
        Args:
            num_classes (int): Number of classes to be classified.
            in_channels (int): Number of channels in input feature.
            init_std (float): Std value for Initiation. Defaults to 0.02.
        """
        super().__init__()

        self.init_std = init_std
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def init_weights(self) -> None:
        """Initiate the parameters from turn normal distribution."""
        trunc_normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x: Tensor) -> Tensor:
        """Forward input Class Token to Linear layer.

        Args:
            x (Tensor): batch of input class tokens.

        Returns:
            Tensor: batch of logits of each class.
        """
        # [N, in_channels]
        cls_score = self.fc_cls(x)

        # [N, num_classes]
        return cls_score

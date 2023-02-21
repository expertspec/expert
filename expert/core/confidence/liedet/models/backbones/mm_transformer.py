from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ..registry import build, registry


@registry.register_module()
class AttentionBottleneckTransformer(nn.Module):
    """Attention Bottleneck Transformer.

    This class implements one of advanced methods to explore multiple modalities separately
    but with sharing information between them.

    See also: `Attention Bottlenecks for Multimodal Fusion`_.

    .. _`Attention Bottlenecks for Multimodal Fusion`: https://arxiv.org/pdf/2107.00135.pdf

    """

    def __init__(
        self,
        transformers: list[dict[str, Any]] | tuple[dict[str, Any]],
        embed_dims: int = 768,
        neck_size: int = 4,
        cls_only: bool = True,
        **kwargs,
    ):
        """
        Args:
            transformers (list[dict[str, Any]] | tuple[dict[str, Any]]): list of
                transformer configs.
            embed_dims (int, optional): size of embedding.
                Defaults to 768.
            neck_size (int, optional): size of bottleneck which is shared between transformers.
                Defaults to 4.
            cls_only (bool, optional): boolean flag to return only class token.
                Otherwise the full tensor with class token and features is returned.
                Defaults to True.
        """
        super().__init__()

        self.transformers = [build(cfg=transformer) for transformer in transformers]
        self.bottleneck = nn.Parameter(data=torch.zeros(1, neck_size, embed_dims))

        self.cls_only = cls_only

    def forward(self, *per_transformer_x: Tensor) -> tuple[Tensor, ...] | Tensor:
        """Forwards input tensors and shared bottleneck to correpsonding transformers.

        It also calculates and stores bottleneck for next iteration
        as mean of bottlenecks after each transformer.

        Each transformer takes same shared bottleneck.

        Args:
            per_transformer_x (tuple[Tensor, ...]): tuple of batch inputs for corresponding transformers.

        Returns:
            tuple[Tensor, ...] | Tensor: if cls_only is True when it returns tuple of tensors of features
            with class token of after each transformer. Otherwise only class token is returned.
        """
        batch_size = per_transformer_x[0].size(0)

        shared_neck = self.bottleneck.expand(batch_size, -1, -1)
        next_shared_neck = torch.zeros(shared_neck.size())

        for x, transformer in zip(per_transformer_x, self.transformers):
            x = torch.cat((x, shared_neck), dim=1)
            x = transformer(x)
            next_shared_neck += x[:, -shared_neck.size(1) :]

            if self.cls_only:
                x = x[:, 0]
            else:
                x = x[:, : -shared_neck.size(1)]

        next_shared_neck /= len(per_transformer_x)
        self.bottleneck.copy_(next_shared_neck)

        return per_transformer_x

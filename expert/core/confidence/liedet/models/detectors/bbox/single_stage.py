from __future__ import annotations

import torch.nn as nn

from mmcv.utils import ConfigDict

from ...registry import build, registry


@registry.register_module(force=True)
class SingleStageDetector(nn.Module):
    def __init__(
        self,
        backbone: nn.Module | ConfigDict,
        bbox_head: nn.Module | ConfigDict,
        neck: nn.Module | ConfigDict | None = None,
    ):
        super().__init__()

        self.backbone = backbone if isinstance(backbone, nn.Module) else build(cfg=backbone, registry=registry)
        self.neck = neck
        if neck is not None:
            self.neck = neck if isinstance(neck, nn.Module) else build(cfg=neck, registry=registry)
        self.bbox_head = bbox_head if isinstance(bbox_head, nn.Module) else build(cfg=bbox_head, registry=registry)

    def forward(self, x):
        h = self.backbone(x)
        if self.neck is not None:
            h = self.neck(h)
        h = self.bbox_head(h)

        return h

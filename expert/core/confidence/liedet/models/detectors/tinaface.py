from collections import OrderedDict
from typing import Any

import torch
from torch import Tensor

from mmcv.parallel import collate

from ..registry import build, registry
from .bbox import SingleStageDetector


@registry.register_module()
class Tinaface(SingleStageDetector):
    """Tinaface single stage detector.

    This model combines ResNet50 backbone, Future Pyramid Network and Inception module.

    See also: `TinaFace: Strong but Simple Baseline for Face Detection`_.

    .. _`TinaFace: Strong but Simple Baseline for Face Detection`: https://arxiv.org/pdf/2011.13183v3.pdf

    """

    def __init__(self, frame_to_result: dict[str, Any], extract_bboxes: dict[str, Any], **kwargs) -> None:
        """
        Args:
            frame_to_result (dict[str, Any]): dictionary of parameters
                for converting model results to bounded boxes.
            extract_bboxes (dict[str, Any]): dictionary of parameters
                for extracting bboxes.
        """

        super().__init__(**kwargs)

        self.frame_to_result = build(cfg=frame_to_result)
        self.extract_bboxes = build(cfg=extract_bboxes)

        # TODO: remove me
        state_dict = torch.load("weights/tinaface_r50_fpn_gn_dcn.pth", map_location="cpu")
        new_state_dict = OrderedDict()
        for old_key, value in state_dict.items():
            new_key = old_key
            if "neck.0" in old_key:
                new_key = old_key.replace("neck.0", "neck.fpn")
            elif "neck.1" in old_key:
                new_key = old_key.replace("neck.1", "neck.inception")
            elif "retina" in old_key:
                new_key = old_key.replace("retina", "conv")
            new_state_dict[new_key] = state_dict[old_key]
        self.load_state_dict(new_state_dict)

    @torch.no_grad()
    def forward(self, x: Tensor) -> tuple[Tensor, ...]:
        """Forwards batch of input images to model, regresses and extracts bboxes.

        Args:
            x (Tensor): batch pf input images.

        Returns:
            tuple[Tensor, ...]: tuple of extracted bboxes.

        .. note::

            The result is tuple due to different sizes of extracted bboxes.

        """
        results = []
        for frame in x:
            h = collate([frame])
            h = super().forward(h)

            h = self.frame_to_result(h)
            results.append(h)

        bboxes = self.extract_bboxes(x, results)

        return bboxes

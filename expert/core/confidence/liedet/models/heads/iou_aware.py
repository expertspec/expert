from __future__ import annotations

from typing import Any

import torch.nn as nn

from mmcv.cnn import ConvModule, bias_init_with_prob, constant_init, normal_init
from mmdet.models.dense_heads import AnchorHead

from ..registry import registry


@registry.register_module()
class IoUAwareRetinaHead(AnchorHead):
    """IoU Aware Retina Head for Single-Stage Object Detector.

    It combines
    - sequence of Convolutional layers to generate logits for target classes
    - sequence of Convolutional layers to regress bboxes
    - three last Convolutional layers to predict scores, bboxes and iou

    See also: `IoU-aware Single-stage Object Detector for Accurate Localization`_.

    .. _`IoU-aware Single-stage Object Detector for Accurate Localization`: https://arxiv.org/abs/1912.05992

    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        stacked_convs: int = 4,
        conv_cfg: dict[str, Any] | None = None,
        norm_cfg: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            num_classes (int): number of target classes
            in_channels (int): number of input features
            stacked_convs (int, optional): number of stacked convolutionals.
                Defaults to 4.
            conv_cfg (dict[str, Any] | None, optional): dictionary with parameters of convolution layers.
                Defaults to None.
            norm_cfg (dict[str, Any] | None, optional): dictionary with parameters for weights initialization.
                Defaults to None.
        """
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        super().__init__(num_classes=num_classes, in_channels=in_channels, **kwargs)

    def _init_layers(self):
        self._init_cls_convs()
        self._init_reg_convs()
        self._init_predictor()

    def _init_cls_convs(self):
        self.cls_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    in_channels=chn,
                    out_channels=self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )

    def _init_reg_convs(self):
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    in_channels=chn,
                    out_channels=self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )

    def _init_predictor(self):
        self.conv_cls = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=self.num_base_priors * self.cls_out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv_reg = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=self.num_base_priors * 4,
            kernel_size=3,
            padding=1,
        )
        self.conv_iou = nn.Conv2d(
            in_channels=self.feat_channels,
            out_channels=self.num_base_priors,
            kernel_size=3,
            padding=1,
        )

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(module=m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(module=m.conv, std=0.01)
        bias_cls = bias_init_with_prob(prior_prob=0.01)
        normal_init(module=self.conv_reg, std=0.01, bias=bias_cls)
        constant_init(module=self.conv_reg, val=0.0)
        constant_init(module=self.conv_iou, val=0.0)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.conv_cls(cls_feat)
        bbox_pred = self.conv_reg(reg_feat)
        iou_pred = self.conv_iou(reg_feat)

        return cls_score, bbox_pred, iou_pred

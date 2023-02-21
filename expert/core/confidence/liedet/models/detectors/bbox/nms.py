from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn

from mmcv.utils import ConfigDict
from mmdet.core.bbox import bbox2result
from mmdet.core.post_processing import multiclass_nms

from ...registry import build, registry


@registry.register_module()
class Frames2Results(nn.Module):
    def __init__(
        self,
        frame_meta,
        meshgrid,
        converter,
        score_thr,
        nms_cfg,
        max_num,
        return_inds=False,
    ):
        super().__init__()

        self.frame_meta = frame_meta

        self.nms_cfg = nms_cfg
        self.max_num = max_num
        self.score_thr = score_thr
        self.return_inds = return_inds

        self.meshgrid = build(cfg=meshgrid, registry=registry)
        self.converter = build(cfg=converter, registry=registry)

    def forward(self, x):
        # x: (cls_score, bbox_pred, ...)

        cls_score = x[0]
        bbox_pred = x[1]

        batch_size = cls_score[0].shape[0]

        featmap_sizes = [featmap.shape[-2:] for featmap in cls_score]
        anchor_mesh = self.meshgrid.gen_anchor_mesh(
            featmap_sizes,
            [self.frame_meta for _ in range(batch_size)],
            cls_score[0].dtype,
            cls_score[0].device,
        )
        dets = self.converter.get_bboxes(
            mlvl_anchors=anchor_mesh,
            img_metas=[self.frame_meta for _ in range(batch_size)],
            cls_scores=cls_score,
            bbox_preds=bbox_pred,
        )

        result_list = []
        for ii in range(batch_size):
            bboxes, scores, centerness = dets[ii]
            det_bboxes, det_labels = multiclass_nms(
                multi_bboxes=bboxes,
                multi_scores=scores,
                score_thr=self.score_thr,
                nms_cfg=self.nms_cfg,
                max_num=self.max_num,
                score_factors=centerness,
                return_inds=self.return_inds,
            )
            bbox_result = bbox2result(det_bboxes, det_labels, 2)
            result_list.append(bbox_result)

        return result_list


@registry.register_module()
class ExtractBBoxes(nn.Module):
    def __init__(self, single: bool = False, no_result_placeholder: str = "full"):
        super().__init__()

        self.single = single
        if no_result_placeholder not in {"full"}:
            raise NotImplementedError
        self.no_result_placeholder = no_result_placeholder

    def forward(self, batch_frames, batch_results):
        batch_bboxes = []
        for frame, results in zip(batch_frames, batch_results):
            bboxes = []
            for result in results:
                try:
                    bbox = result[0][0][:4].astype(np.int32)
                    extracted = frame[:, bbox[1] : bbox[3], bbox[0] : bbox[2]]
                except IndexError:
                    extracted = frame
                if self.single:
                    break
                bboxes.append(extracted)

            if self.single and len(bboxes) > 0:
                batch_bboxes.append(bboxes[0])
            elif self.single:
                if self.no_result_placeholder == "full":
                    batch_bboxes.append(frame)
            elif len(bboxes) > 0:
                batch_bboxes.append(bboxes)
            else:
                if self.no_result_placeholder == "full":
                    batch_bboxes.append([frame])

        return torch.stack(batch_bboxes)

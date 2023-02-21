
from __future__ import annotations

from catalyst import dl
from einops import rearrange

import torch
import torch.nn as nn
from torch import Tensor

from mmcv.cnn.utils.weight_init import kaiming_init, trunc_normal_
from mmcv.utils import ConfigDict

from .base_module import BaseModule
from .registry import build, registry


@registry.register_module()
class LieDetector(BaseModule):
    """Lie Detector Pipeline.

    This class combines video and audio feature extractors.
    After features extraction it concatenates features, add class token
    and add position embedding.
    Then it forwards them into time model to extract time dependent features.
    At the end enriched class token is forwarded to classification header
    which returns logits over target classes.
    """

    def __init__(
        self,
        *,
        time_model: nn.Module | dict,
        cls_head: nn.Module | dict,
        features_dims: int | None = None,
        embed_dims: int | None = None,
        video_model: nn.Module | dict | None = None,
        audio_model: nn.Module | dict | None = None,
        window: int = 100,
        pos_embed: bool = True,
        init: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            time_model (nn.Module | dict): model to extract time-dependent features
                or configuration dictionary to build it.
            cls_head (nn.Module | dict): classification head or configuration dictionary
                to build it.
            features_dims (int | None, optional): size of features to embed.
                If `None` when raw features are used, otherwise features are embedded.
                Defaults to None.
            embed_dims (int | None, optional): size of features embedding.
                If `None` when features are not embedded,
                otherwise if `features_dims` also is not `None`
                then features are embedded to this size.
                features_dims --[Projection]--> embed_dims
                Defaults to None.
            video_model (nn.Module | dict | None, optional): model to extract features
                from sequence of images (video) or configuration dictionary to build it.
                Defaults to None.
            audio_model (nn.Module | dict | None, optional): model to extract features
                from audio sequence or configuration dictionary to build it.
                Defaults to None.
            window (int, optional): number of video frames per window.
                Defaults to 100.
            pos_embed (bool, optional): boolean flag to use position embedding.
                Mostly uses with Transformer time model.
                Defaults to True.
            init (bool, optional): boolean flag to initialize weights or load pretrained ones from config.
                Defaults to True.
        """
        super().__init__(init=False, **kwargs)

        if video_model is None and audio_model is None:
            raise ValueError("At least one of video_model or audio_model should be specified.")

        self.video_model = video_model
        self.audio_model = audio_model
        self.time_model = time_model
        self.cls_head = cls_head

        self.window = window

        if isinstance(self.video_model, ConfigDict):
            self.video_model = build(cfg=self.video_model, registry=registry)
        if isinstance(self.audio_model, ConfigDict):
            self.audio_model = build(cfg=self.audio_model, registry=registry)
        if isinstance(self.time_model, ConfigDict):
            self.time_model = build(cfg=self.time_model, registry=registry)
        if isinstance(self.cls_head, ConfigDict):
            self.cls_head = build(cfg=self.cls_head, registry=registry)

        if features_dims is not None and embed_dims is not None:
            self.use_embed = True
            self.embed = nn.Linear(in_features=features_dims, out_features=embed_dims)
        else:
            self.use_embed = False
            self.embed = nn.Identity()

        if self.use_embed:
            self.cls_tokens = nn.Parameter(torch.zeros(1, 1, embed_dims))
            self.pos_embed = pos_embed
            self.pos_embeds = nn.Parameter(torch.zeros(1, window + 1, embed_dims))

        if init:
            self.init_weights()

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Forwards batch of video/audio frames over modules.

        Args:
            batch (dict[str, Tensor]): dictionary with batch of video and audio sequences.
                Should contains `video_frames` and `audio_frames`.

        Returns:
            Tensor: batch of logits.
        """

        vframes, aframes = batch["video_frames"], batch["audio_frames"]

        if self.video_model is not None:
            vfeatures, is_face = self.video_model(vframes)
            if sum(is_face) == 0:
                return 'No face'
        if self.audio_model is not None:
            afeatures = self.audio_model(aframes)

        if self.video_model is not None and self.audio_model is not None:
            logits = torch.cat((vfeatures, afeatures), dim=-1)
            del vfeatures, afeatures
        elif self.video_model is not None:
            logits = vfeatures
        else:
            logits = afeatures

        logits = self.embed(logits)

        shape_size = len(logits.shape)

        if self.use_embed:
            if shape_size == 3:
                cls_tokens = self.cls_tokens.expand(logits.size(0), -1, -1)
                logits = torch.cat((logits, cls_tokens), dim=1)

            if shape_size == 3 and self.pos_embed:
                logits = logits + self.pos_embeds

        logits = self.time_model(logits)

        if self.use_embed:
            if shape_size == 3:
                logits = logits[:, -1]

        logits = self.cls_head(logits)

        return logits

    def init_weights(self) -> None:
        """Recursively initializes weights of modules from configs."""
        if self.use_embed:
            trunc_normal_(self.pos_embeds, std=0.02)
            trunc_normal_(self.cls_tokens, std=0.02)

        if isinstance(self.embed, nn.Linear):
            kaiming_init(self.embed, mode="fan_in", nonlinearity="linear")

        super().init_weights()


class LieDetectorRunner(dl.Runner):
    """Base Lie Detector Runner."""

    @torch.no_grad()
    def predict_batch(self, batch: dict[str, Tensor]) -> Tensor:
        """Predicts indexes of target classes over batch.

        Args:
            batch (dict[str, Tensor]): dictionary with batch of video and audio sequences.

        Returns:
            Tensor: batch of target class indexes.
        """
        self.model.eval()
        logits = self.model(batch)
        if isinstance(logits, str):
            return 'No face'
        probs = torch.sigmoid(logits)

        return probs

    def predict_sample(self, sample: dict[str, Tensor]) -> Tensor:
        """Predicts index of target classes over single sample.

        Args:
            sample (dict[str, Tensor]): dictionary with video and audio sequences.

        Returns:
            Tensor: target class index.
        """
        sample["video_frames"] = rearrange(sample["video_frames"], "(b t) c h w -> b t c h w", b=1)
        sample["audio_frames"] = rearrange(sample["audio_frames"], "(b c) t -> b c t", b=1)

        return self.predict_batch(sample)

    def handle_batch(self, batch: dict[str, Tensor]) -> None:
        _, _, labels = batch["video_frames"], batch["audio_frames"], batch["labels"]

        logits = self.model(batch)

        self.batch = dict(logits=logits, labels=labels)

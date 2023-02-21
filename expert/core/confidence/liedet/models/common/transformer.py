from __future__ import annotations

from typing import Any

from einops import rearrange

import torch
import torch.nn as nn
from torch import Tensor

from mmcv.cnn import build_norm_layer, constant_init
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.cnn.builder import build_from_cfg
from mmcv.runner.base_module import BaseModule
from mmcv.utils import digit_version

from ..registry import registry


@registry.register_module(force=True)
class TransformerEncoder(nn.TransformerEncoder):
    """Wrapper of torch TransformerEncoder.

    The class allows to create transformer from config file.
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        encoder_layer = build_from_cfg(cfg=encoder_layer, registry=registry)
        norm = build_from_cfg(cfg=norm, registry=registry) if norm is not None else None

        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers, norm=norm)


class DividedTemporalAttentionWithNorm(BaseModule):
    """Temporal Attention in Divided Space Time Attention."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        num_frames: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dropout_layer: dict[str, Any] = dict(type="DropPath", drop_prob=0.1),
        norm_cfg: dict[str, Any] = dict(type="LN"),
        init_cfg: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Args:
            embed_dims (int): Dimensions of embedding.
            num_heads (int): Number of parallel attention heads in
                TransformerCoder.
            num_frames (int): Number of frames in the video.
            attn_drop (float): A Dropout layer on attn_output_weights. Defaults to
                0.0.
            proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
                Defaults to 0.0.
            dropout_layer (dict): The dropout_layer used when adding the shortcut.
                Defaults to `dict(type='DropPath', drop_prob=0.1)`.
            norm_cfg (dict): Config dict for normalization layer. Defaults to
                `dict(type='LN')`.
            init_cfg (dict | None): The Config for initialization. Defaults to
                None.
        """
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

        if digit_version(torch.__version__) < digit_version("1.9.0"):
            kwargs.pop("batch_first", None)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()
        self.temporal_fc = nn.Linear(self.embed_dims, self.embed_dims)

        self.init_weights()

    def init_weights(self) -> None:
        """Constant weights initialization of temporal linear layer."""
        constant_init(self.temporal_fc, val=0, bias=0)

    def forward(self, query: Tensor, key=None, value=None, residual=None, **kwargs) -> Tensor:
        """Forwards Divided Temporal Attention with Normalization

        (Input)--[Extract Class Token]-->(Initial Class Token)

        (Input without Class Token)--
            --[Normalization]--
            --[Temporal Attention]--
            --[Dropout]--
            --[Temporal Linear]--
            --[Identity]--
            --[Add Initial Class Token]--
        -->(Output)

        Args:
            query (Tensor): input time sequence.

        Returns:
            Tensor: output time sequence.
        """
        assert residual is None, "Always adding the shortcut in the forward function"

        init_cls_token = query[:, 0, :].unsqueeze(1)
        identity = query_t = query[:, 1:, :]

        # query_t [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_t.size()
        p, t = pt // self.num_frames, self.num_frames

        # res_temporal [batch_size * num_patches, num_frames, embed_dims]
        query_t = self.norm(query_t.reshape(b * p, t, m)).permute(1, 0, 2)
        res_temporal = self.attn(query_t, query_t, query_t)[0].permute(1, 0, 2)
        res_temporal = self.dropout_layer(self.proj_drop(res_temporal.contiguous()))
        res_temporal = self.temporal_fc(res_temporal)

        # res_temporal [batch_size, num_patches * num_frames, embed_dims]
        res_temporal = res_temporal.reshape(b, p * t, m)

        # ret_value [batch_size, num_patches * num_frames + 1, embed_dims]
        new_query_t = identity + res_temporal
        new_query = torch.cat((init_cls_token, new_query_t), 1)

        return new_query


class DividedSpatialAttentionWithNorm(BaseModule):
    """Spatial Attention in Divided Space Time Attention."""

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        num_frames: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        dropout_layer: dict[str, Any] = dict(type="DropPath", drop_prob=0.1),
        norm_cfg: dict[str, Any] = dict(type="LN"),
        init_cfg: dict[str, Any] | None = None,
        **kwargs,
    ):
        """
        Args:
            embed_dims (int): Dimensions of embedding.
            num_heads (int): Number of parallel attention heads in TransformerCoder.
            num_frames (int): Number of frames in the video.
            attn_drop (float): A Dropout layer on attn_output_weights.
                Defaults to 0.0.
            proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
                Defaults to 0.0.
            dropout_layer (dict): The dropout_layer used when adding the shortcut.
                Defaults to `dict(type='DropPath', drop_prob=0.1)`.
            norm_cfg (dict): Config dict for normalization layer.
                Defaults to `dict(type='LN')`.
            init_cfg (dict | None): The Config for initialization.
                Defaults to None.
        """
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]
        if digit_version(torch.__version__) < digit_version("1.9.0"):
            kwargs.pop("batch_first", None)
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop, **kwargs)
        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()

        self.init_weights()

    def init_weights(self) -> None:
        # init DividedSpatialAttentionWithNorm by default
        pass

    def forward(self, query: Tensor, key=None, value=None, residual=None, **kwargs) -> Tensor:
        """Forwards Divided Spatial Attention with Normalization.

        (Input)--
            --[Expand Class Token over Spatial dim]--
            --[Normalization]--
            --[Spatial Attention]--
            --[Dropout]--
            --[Spatial Linear]--
            --[Mean over expanded Class Tokens]--
            --[Identity]--
        -->(Output)

        Args:
            query (Tensor): input time sequence.

        Returns:
            Tensor: output time sequence.
        """

        assert residual is None, "Always adding the shortcut in the forward function"

        identity = query
        init_cls_token = query[:, 0, :].unsqueeze(1)
        query_s = query[:, 1:, :]

        # query_s [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_s.size()
        p, t = pt // self.num_frames, self.num_frames

        # cls_token [batch_size * num_frames, 1, embed_dims]
        cls_token = init_cls_token.repeat(1, t, 1).reshape(b * t, m).unsqueeze(1)

        # query_s [batch_size * num_frames, num_patches + 1, embed_dims]
        query_s = rearrange(query_s, "b (p t) m -> (b t) p m", p=p, t=t)
        query_s = torch.cat((cls_token, query_s), 1)

        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        query_s = self.norm(query_s).permute(1, 0, 2)
        res_spatial = self.attn(query_s, query_s, query_s)[0].permute(1, 0, 2)
        res_spatial = self.dropout_layer(self.proj_drop(res_spatial.contiguous()))

        # cls_token [batch_size, 1, embed_dims]
        cls_token = res_spatial[:, 0, :].reshape(b, t, m)
        cls_token = torch.mean(cls_token, 1, True)

        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        res_spatial = rearrange(res_spatial[:, 1:, :], "(b t) p m -> b (p t) m", p=p, t=t)
        res_spatial = torch.cat((cls_token, res_spatial), 1)

        new_query = identity + res_spatial
        return new_query


class FFNWithNorm(FFN):
    """FFN with pre normalization layer.

    It apply one normalization layer before forwarding the input data to feed-forward networks.
    """

    def __init__(self, *args, norm_cfg: dict[str, Any] = dict(type="LN"), **kwargs) -> None:
        """
        Args:
            embed_dims (int): Dimensions of embedding.
                Defaults to 256.
            feedforward_channels (int): Hidden dimension of FFNs.
                Defaults to 1024.
            num_fcs (int, optional): Number of fully-connected layers in FFNs.
                Defaults to 2.
            act_cfg (dict): Config for activate layers.
                Defaults to `dict(type='ReLU')`
            ffn_drop (float, optional): Probability of an element to be zeroed in FFN.
                Defaults to 0..
            add_residual (bool, optional): Whether to add the residual connection.
                Defaults to `True`.
            dropout_layer (dict | None): The dropout_layer used when adding the shortcut.
                Defaults to None.
            init_cfg (dict | None): The Config for initialization.
                Defaults to None.
            norm_cfg (dict): Config dict for normalization layer.
                Defaults to `dict(type='LN')`.
        """

        super().__init__(*args, **kwargs)

        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

    def forward(self, x: Tensor, residual=None) -> Tensor:
        """Applies normalization and then forwards the input data to feed-forward networks.

        Args:
            x (Tensor): input tensor.

        Returns:
            Tensor: output tensor.
        """

        assert residual is None, "Cannot apply pre-norm with FFNWithNorm"

        return super().forward(self.norm(x), x)

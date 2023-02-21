from __future__ import annotations

from typing import overload

from einops import rearrange

import torch.nn as nn
from torch import Tensor

from ..registry import registry


def select(x, index=None, key=None):
    if index is None and key is None:
        raise ValueError

    if index is not None:
        return x[index]
    key = dict.fromkeys(key)

    return {k: v for k, v in x.items() if k in key}


@registry.register_module()
class Select(nn.Module):
    @overload
    def __init__(self, index: int | tuple[int, ...] | list[int]) -> None:
        ...  # noqa: WPS428

    @overload
    def __init__(self, key: str | tuple[str, ...] | list[str]) -> None:
        ...  # noqa: WPS428

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.kwargs = kwargs

    def forward(self, x) -> Tensor:
        return select(x, **self.kwargs)


@registry.register_module()
class Rearrange(nn.Module):
    def __init__(self, pattern: str, **axes_length: int):
        super().__init__()

        self.pattern = pattern
        self.axes_length = axes_length

    def forward(self, x):
        return rearrange(x, self.pattern, **self.axes_length)

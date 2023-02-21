from __future__ import annotations

from typing import Optional, Union, overload

import torch
import torch.nn as nn
from torch import Tensor

from ..models.registry import registry


@registry.register_module()
class Concat(nn.Module):
    @overload
    def __init__(self, dim: int = 0, *, out: Optional[Tensor] = None) -> None:
        ...  # noqa: WPS428

    @overload
    def __init__(self, dim: Union[str, None], *, out: Optional[Tensor] = None) -> None:
        ...  # noqa: WPS428

    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.kwargs = kwargs

    def forward(self, *x) -> Tensor:
        return torch.cat(x, **self.kwargs)


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
class Reshape(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.kwargs = kwargs

    def forward(self, x) -> Tensor:
        return x.reshape(**self.kwargs)


@registry.register_module()
class Permute(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self.kwargs = kwargs

    def forward(self, x) -> Tensor:
        return x.permute(**self.kwargs)

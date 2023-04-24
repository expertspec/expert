from __future__ import annotations

import logging
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor, device


def constant_init(
    module: nn.Module, val: int | float, bias: int | float = 0
) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(tensor=module.weight, val=val)  # type: ignore
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(tensor=module.bias, val=bias)  # type: ignore


def normal_init(
    module: nn.Module,
    mean: int | float = 0,
    std: int | float = 1,
    bias: int | float = 0,
) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(tensor=module.weight, mean=mean, std=std)  # type: ignore
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(tensor=module.bias, val=bias)  # type: ignore


def uniform_init(
    module: nn.Module,
    a: int | float = 0,
    b: int | float = 1,
    bias: int | float = 0,
) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.uniform_(tensor=module.weight, a=a, b=b)  # type: ignore
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(tensor=module.bias, bias=bias)  # type: ignore


def xavier_init(
    module: nn.Module,
    gain: int | float = 1,
    bias: int | float = 0,
    distribution: str = "normal",
) -> None:
    supported_distributions = {"normal", "uniform"}
    if distribution not in supported_distributions:
        raise NotImplementedError(
            f"Distribution {distribution} is unknown. Supported distributions: {supported_distributions}"
        )

    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.xavier_uniform_(tensor=module.weight, gain=gain)  # type: ignore
        else:
            nn.init.xavier_normal_(tensor=module.weight, gain=gain)  # type: ignore
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(tensor=module.bias, val=bias)  # type: ignore


def kaiming_init(
    module: nn.Module,
    a: int | float = 0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: int | float = 0,
    distribution: str = "normal",
) -> None:
    supported_distributions = {"normal", "uniform"}
    if distribution not in supported_distributions:
        raise NotImplementedError(
            f"Distribution {distribution} is unknown. Supported distributions: {supported_distributions}"
        )

    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(
                tensor=module.weight,
                a=a,
                mode=mode,
                nonlinearity=nonlinearity,
            )
        else:
            nn.init.kaiming_normal_(
                tensor=module.weight,
                a=a,
                mode=mode,
                nonlinearity=nonlinearity,
            )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(tensor=module.bias, val=bias)  # type: ignore


def caffe2_xavier_init(module: nn.Module, bias: int | float = 0) -> None:
    kaiming_init(
        module=module,
        a=1,
        mode="fan_in",
        nonlinearity="leaky_relu",
        bias=bias,
        distribution="uniform",
    )


def bias_init_with_prob(prior_prob: float) -> float:
    bias_init = float(-np.log((1 - prior_prob) / prior_prob))

    return bias_init


def _no_grad_trunc_normal_(
    tensor: Tensor,
    mean: int | float = 0,
    std: int | float = 1,
    a: int | float = -2,
    b: int | float = 2,
) -> Tensor:
    def norm_cdf(x: float) -> float:
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            message="mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )
    with torch.no_grad():
        lower = norm_cdf(x=(a - mean) / std)
        upper = norm_cdf(x=(b - mean) / std)

        tensor.uniform_(2 * lower - 1, 2 * upper - 1)
        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)

        return tensor


def trunc_normal_(
    tensor: Tensor,
    mean: int | float = 0,
    std: int | float = 1,
    a: int | float = -2,
    b: int | float = 2,
) -> Tensor:
    return _no_grad_trunc_normal_(tensor=tensor, mean=mean, std=std, a=a, b=b)


def trunc_normal_init(
    module: nn.Module,
    mean: int | float = 0,
    std: int | float = 1,
    a: int | float = -2,
    b: int | float = 2,
    bias: int | float = 0,
) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        trunc_normal_(tensor=module.weight, mean=mean, std=std, a=a, b=b)  # type: ignore
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(tensor=module.bias, val=bias)  # type: ignore


class BaseInit(object):
    def __init__(
        self,
        *,
        bias: int | float = 0,
        bias_prob: float | None = None,
        layer: str | list[str] | tuple[str, ...] | None = None,
        **kwargs,
    ) -> None:
        self.wholemodule = False

        if not isinstance(bias, (int, float)):
            raise TypeError(f"bias must be a number, but got a {type(bias)}")

        if bias_prob is not None and not isinstance(bias_prob, float):
            raise TypeError(
                f"bias_prob must be float, but got a {type(bias_prob)}"
            )

        if layer is not None and not isinstance(layer, (str, list, tuple)):
            raise TypeError(
                f"layer must be a str or a list/tuple or str, but got a {type(layer)}"
            )
        elif layer is None:
            self.wholemodule = True
            layer = []
        elif isinstance(layer, str):
            layer = [layer]
        self.layer = layer

        if bias_prob is not None:
            self.bias = bias_init_with_prob(prior_prob=bias_prob)
        else:
            self.bias = bias

    def __call__(self, module: nn.Module) -> None:
        raise NotImplementedError

    def _get_init_info(self) -> str:
        info = f"{self.__class__.__name__}, bias={self.bias}"
        if self.layer is not None:
            info = f"{info}, layer={self.layer}"

        return info

    def __repr__(self) -> str:
        return self._get_init_info()


def _get_bases_names(module: nn.Module) -> list[str]:
    return [b.__name__ for b in module.__class__.__bases__]


class ConstantInit(BaseInit):
    def __init__(self, val: int | float, **kwargs) -> None:
        super().__init__(**kwargs)

        self.val = val

    def __call__(self, module: nn.Module) -> None:
        def init(m: nn.Module) -> None:
            if self.wholemodule:
                constant_init(module=m, val=self.val, bias=self.bias)
            else:
                layername = m.__class__.__name__
                basesnames = _get_bases_names(module=m)
                if len(set(self.layer) & set([layername] + basesnames)):
                    constant_init(module=m, val=self.val, bias=self.bias)

        module.apply(init)

    def _get_init_info(self) -> str:
        info = f"{self.__class__.__name__}: val={self.val}, bias={self.bias}"
        if self.layer is not None:
            info = f"{info}, layer={self.layer}"

        return info


class NormalInit(BaseInit):
    def __init__(
        self, mean: int | float = 0, std: int | float = 1, **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.mean = mean
        self.std = std

    def __call__(self, module: nn.Module) -> None:
        def init(m: nn.Module) -> None:
            if self.wholemodule:
                normal_init(m, self.mean, self.std, self.bias)
            else:
                layername = m.__class__.__name__
                basesnames = _get_bases_names(module=m)
                if len(set(self.layer) & set([layername] + basesnames)):
                    normal_init(
                        module=m, mean=self.mean, std=self.std, bias=self.bias
                    )

        module.apply(init)

    def _get_init_info(self) -> str:
        info = (
            f"{self.__class__.__name__}: mean={self.mean},"
            f" std={self.std}, bias={self.bias}"
        )
        if self.layer is not None:
            info = f"{info}, layer={self.layer}"

        return info


class UniformInit(BaseInit):
    def __init__(self, a: int | float = 0, b: int | float = 1, **kwargs):
        super().__init__(**kwargs)

        self.a = a
        self.b = b

    def __call__(self, module: nn.Module) -> None:
        def init(m: nn.Module) -> None:
            if self.wholemodule:
                uniform_init(module=m, a=self.a, b=self.b, bias=self.bias)
            else:
                layername = m.__class__.__name__
                basesnames = _get_bases_names(m)
                if len(set(self.layer) & set([layername] + basesnames)):
                    uniform_init(module=m, a=self.a, b=self.b, bias=self.bias)

        module.apply(init)

    def _get_init_info(self) -> str:
        info = (
            f"{self.__class__.__name__}: a={self.a},"
            f" b={self.b}, bias={self.bias}"
        )
        if self.layer is not None:
            info = f"{info}, layer={self.layer}"

        return info


class XavierInit(BaseInit):
    def __init__(
        self, gain: int | float = 1, distribution: str = "normal", **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.gain = gain
        self.distribution = distribution

    def __call__(self, module: nn.Module) -> None:
        def init(m: nn.Module) -> None:
            if self.wholemodule:
                xavier_init(
                    module=m,
                    gain=self.gain,
                    bias=self.bias,
                    distribution=self.distribution,
                )
            else:
                layername = m.__class__.__name__
                basesnames = _get_bases_names(m)
                if len(set(self.layer) & set([layername] + basesnames)):
                    xavier_init(
                        module=m,
                        gain=self.gain,
                        bias=self.bias,
                        distribution=self.distribution,
                    )

        module.apply(init)

    def _get_init_info(self) -> str:
        info = (
            f"{self.__class__.__name__}: gain={self.gain}, "
            f"distribution={self.distribution}, bias={self.bias}"
        )
        if self.layer is not None:
            info = f"{info}, layer={self.layer}"

        return info


class KaimingInit(BaseInit):
    def __init__(
        self,
        a: int | float = 0,
        mode: str = "fan_out",
        nonlinearity: str = "relu",
        distribution: str = "normal",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.a = a
        self.mode = mode
        self.nonlinearity = nonlinearity
        self.distribution = distribution

    def __call__(self, module: nn.Module) -> None:
        def init(m: nn.Module) -> None:
            if self.wholemodule:
                kaiming_init(
                    module=m,
                    a=self.a,
                    mode=self.mode,
                    nonlinearity=self.nonlinearity,
                    bias=self.bias,
                    distribution=self.distribution,
                )
            else:
                layername = m.__class__.__name__
                basesnames = _get_bases_names(m)
                if len(set(self.layer) & set([layername] + basesnames)):
                    kaiming_init(
                        module=m,
                        a=self.a,
                        mode=self.mode,
                        nonlinearity=self.nonlinearity,
                        bias=self.bias,
                        distribution=self.distribution,
                    )

        module.apply(init)

    def _get_init_info(self) -> str:
        info = (
            f"{self.__class__.__name__}: a={self.a}, mode={self.mode}"
            f", nonlinearity={self.nonlinearity}"
            f", distribution={self.distribution},"
            f", bias={self.bias}"
        )
        if self.layer is not None:
            info = f"{info}, layer={self.layer}"

        return info


class Caffe2XavierInit(KaimingInit):
    def __init__(self, **kwargs) -> None:
        super().__init__(
            a=1,
            mode="fan_in",
            nonlinearity="leaky_relu",
            distribution="uniform",
            **kwargs,
        )

    def __call__(self, module: nn.Module) -> None:
        super().__call__(module=module)


class TruncNormalInit(BaseInit):
    def __init__(
        self,
        mean: int | float = 0,
        std: int | float = 1,
        a: int | float = -2,
        b: int | float = 2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.mean = mean
        self.std = std
        self.a = a
        self.b = b

    def __call__(self, module: nn.Module) -> None:
        def init(m: nn.Module) -> None:
            if self.wholemodule:
                trunc_normal_init(
                    module=m,
                    mean=self.mean,
                    std=self.std,
                    a=self.a,
                    b=self.b,
                    bias=self.bias,
                )
            else:
                layername = m.__class__.__name__
                basesnames = _get_bases_names(m)
                if len(set(self.layer) & set([layername] + basesnames)):
                    trunc_normal_init(
                        module=m,
                        mean=self.mean,
                        std=self.std,
                        a=self.a,
                        b=self.b,
                        bias=self.bias,
                    )

        module.apply(init)

    def _get_init_info(self) -> str:
        info = (
            f"{self.__class__.__name__}: a={self.a}, b={self.b},"
            f" mean={self.mean}, std={self.std}, bias={self.bias}"
        )
        return info


class PretrainedInit(object):
    def __init__(
        self,
        checkpoint: str,
        prefix: str | None = None,
        prefix_add: bool = False,
        map_location: str | device | None = None,
        strict: bool = False,
        **kwargs,
    ) -> None:
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.prefix_add = prefix_add
        self.map_location = map_location
        self.strict = strict

    def __call__(self, module: nn.Module) -> None:
        logger = logging.getLogger(name="module_init")

        logger.info(
            f"load model from: {self.checkpoint}, to {self.map_location}"
        )

        state_dict = torch.load(
            f=self.checkpoint, map_location=self.map_location
        )

        if not isinstance(state_dict, dict):
            raise TypeError(
                f"Checkpoint {self.checkpoint} should contains dict, but got {type(state_dict)}"
            )

        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]

        if self.prefix is not None:
            prefix = self.prefix
            if not prefix.endswith("."):
                prefix = f"{prefix}."
            prefix_size = len(prefix)

            if self.prefix_add:
                state_dict = {f"{prefix}{k}": v for k, v in state_dict.items()}
            else:
                state_dict = {
                    k[prefix_size:]: v
                    for k, v in state_dict.items()
                    if k.startswith(prefix)
                }

        module.load_state_dict(state_dict=state_dict, strict=True)

    def _get_init_info(self) -> str:
        info = f"{self.__class__.__name__}: load from {self.checkpoint}"

        return info


def initialize(
    module: nn.Module,
    init_cfg: BaseInit | tuple[BaseInit, ...] | list[BaseInit] | None,
) -> None:
    if init_cfg is not None:
        if not isinstance(init_cfg, (tuple, list)):
            init_cfg = [init_cfg]

        for cfg in init_cfg:
            cfg(module=module)

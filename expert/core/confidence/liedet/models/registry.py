from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Optional

from mmaction.models.builder import MODELS as MMACTION_MODELS

import torch.nn as nn
from torch.nn import modules as TORCH_MODULES  # noqa: N812
from torchvision.transforms import transforms as TORCHVISION_TRANSFORMS  # noqa: N812

from mmcv.cnn import MODELS
from mmcv.cnn.bricks.registry import (
    ACTIVATION_LAYERS,
    ATTENTION,
    CONV_LAYERS,
    DROPOUT_LAYERS,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
    PADDING_LAYERS,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    TRANSFORMER_LAYER,
    TRANSFORMER_LAYER_SEQUENCE,
    UPSAMPLE_LAYERS,
)
from mmcv.cnn.bricks.wrappers import Linear, MaxPool2d, MaxPool3d
from mmcv.utils import ConfigDict, Registry
from mmcv.utils.logging import get_logger, logger_initialized
from mmdet.models.builder import MODELS as MMDET_MODELS

from .builder import recursive_build

registry = Registry(name="registry", parent=MODELS)

for m in TORCH_MODULES.__all__:
    module = getattr(TORCH_MODULES, m)
    registry.register_module(name=module.__name__, module=module)

for m in TORCHVISION_TRANSFORMS.__all__:
    module = getattr(TORCHVISION_TRANSFORMS, m)
    registry.register_module(name=module.__name__, module=module)


registries = [
    ACTIVATION_LAYERS,
    ATTENTION,
    CONV_LAYERS,
    DROPOUT_LAYERS,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
    PADDING_LAYERS,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    TRANSFORMER_LAYER,
    TRANSFORMER_LAYER_SEQUENCE,
    UPSAMPLE_LAYERS,
    MMDET_MODELS,
    MMACTION_MODELS,
]

for reg in registries:
    if reg is None:
        continue

    for module_name in reg._module_dict:  # noqa: WPS437
        # May fail on CUDA ops
        try:
            registry.register_module(module_name, force=True, module=reg.get(module_name))
        except ImportError:
            pass

modules = [Linear, MaxPool2d, MaxPool3d]
for module in modules:  # noqa: WPS440
    registry.register_module(module.__name__, force=True, module=module)


def build(
    cfg: dict | list | tuple,
    module_name: Optional[str] = None,  # noqa: WPS442
    registry: Registry = registry,  # noqa: WPS442
    input_key: str = "inputs",
    target_key: str = "logits",
    init: bool = True,
    print_init_info: bool = False,
) -> nn.Module:
    if isinstance(cfg, (list, tuple)):
        cfg_dict = {f"{name}": sub_cfg for name, sub_cfg in enumerate(cfg)}
    elif isinstance(cfg, dict):
        cfg_dict = cfg
    else:
        raise ValueError(f"Config should be dict, list or tuple, but got {type(cfg)}")

    init = cfg_dict.pop("init", init)
    print_init_info = cfg_dict.pop("print_init_info", print_init_info)

    module = recursive_build(
        cfg=cfg_dict,
        module_name=module_name,
        registry=registry,
        input_key=input_key,
        target_key=target_key,
    )

    if init:
        logger_names = list(logger_initialized.keys())
        logger_name = logger_names[0] if logger_names else "mmcv"
        logger = get_logger(name=logger_name)

        # The MMCV logger is too verbose during initialization
        # We suppress it by default during initialization,
        #   but it can be un-suppresed by setting print_init_info=True
        if not print_init_info:
            prev_level = logger.level
            if prev_level < 30:
                logger.setLevel("WARNING")

        if not isinstance(module, nn.Parameter) and hasattr(module, "init_weights"):
            module.init_weights()

        if not print_init_info:
            if prev_level < 30:
                logger.setLevel(prev_level)

    return module


registry.build_func = build


@registry.register_module(force=True)
class Sequential(nn.Sequential):
    def __init__(self, *args, modules: list | None = None):
        super().__init__()

        if modules is not None:
            args = modules

        if len(args) == 1 and isinstance(args[0], (OrderedDict, ConfigDict)):
            for key, module in args[0].items():
                if isinstance(module, dict):
                    module = build(cfg=module, registry=registry)
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                if isinstance(module, dict):
                    module = build(cfg=module, registry=registry)
                self.add_module(str(idx), module)

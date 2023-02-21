from __future__ import annotations

import logging
from abc import ABCMeta

import torch.nn as nn

from liedet.models.utils.weights_init import BaseInit, PretrainedInit, initialize


class BaseModule(nn.Module, metaclass=ABCMeta):
    def __init__(self, init_cfg: BaseInit | tuple[BaseInit, ...] | None = None, init: bool = False, **kwargs) -> None:
        super().__init__()

        self._is_init = False
        self.init_cfg = init_cfg

        if init:
            self.init_weights()

    @property
    def is_init(self) -> bool:
        return self._is_init

    def init_weights(self) -> None:
        is_top_level_module = False

        module_name = self.__class__.__name__
        logger = logging.getLogger(name="module_init")
        if not self._is_init:
            if self.init_cfg is not None:
                logger.info(f"initialize {module_name} with init_cfg {self.init_cfg}")
                initialize(module=self, init_cfg=self.init_cfg)

                if isinstance(self.init_cfg, PretrainedInit):
                    return

            for m in self.children():
                if hasattr(m, "is_init"):
                    if not m.is_init and hasattr(m, "init_weights"):
                        m.init_weights()  # type: ignore
                    elif not m.is_init and m.init_cfg is not None:
                        initialize(module=m, init_cfg=self.init_cfg)
                        initialize(module=m, init_cfg=m.init_cfg)  # type: ignore
                    elif not m.is_init:
                        initialize(module=m, init_cfg=self.init_cfg)
                else:
                    initialize(m, init_cfg=self.init_cfg)
            self._is_init = True
        else:
            logger.warn(f"init_weights of {self.__class__.__name__} has been called more than ones")

        if is_top_level_module:
            for sub_module in self.modules():
                del sub_module._params_init_info

    def __repr__(self) -> str:
        s = super().__repr__()
        if self.init_cfg is not None:
            s += f"\ninit_cfg={self.init_cfg}"

        return s


class Sequential(BaseModule, nn.Sequential):
    def __init__(self, *modules: nn.Module, init_cfg=None, init: bool = False, **kwargs) -> None:
        BaseModule.__init__(self, init_cfg=init_cfg, init=False)
        nn.Sequential.__init__(self, *modules)

        if init:
            self.init_weights()


class ModuleList(BaseModule, nn.ModuleList):
    def __init__(self, modules=None, init_cfg=None, init: bool = False, **kwargs) -> None:
        BaseModule.__init__(self, init_cfg=init_cfg, init=False)
        nn.ModuleList.__init__(self, modules=modules)

        if init:
            self.init_weights()


class ModuleDict(BaseModule, nn.ModuleDict):
    def __init__(self, modules=None, init_cfg=None, init: bool = False, **kwargs) -> None:
        BaseModule.__init__(self, init_cfg=init_cfg, init=False)
        nn.ModuleDict.__init__(self, modules=modules)

        if init:
            self.init_weights()

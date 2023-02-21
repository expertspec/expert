from __future__ import annotations

from typing import OrderedDict

import torch.nn as nn

from .builder import BaseModule
from .registry import build, registry


@registry.register_module()
class Pipeline(BaseModule):
    def __init__(
        self,
        input_key: str = "inputs",
        target_key: str = "logits",
        builded_modules: OrderedDict | None = None,
        **modules,
    ):
        super().__init__()

        if builded_modules is not None:
            modules = builded_modules

        for module_name, cfg in modules.items():
            if isinstance(cfg, nn.Module):
                module = cfg
            else:
                module = build(cfg=cfg, input_key=input_key, target_key=target_key)
            self.add_module(name=module_name, module=module)

        self.input_key = input_key
        self.target_key = target_key

    def forward(self, batch):
        for param_name, param in self.named_parameters():
            if param_name.startswith("param_"):
                batch[param_name] = param

        return self.forward_next(batch=batch, module=self)

    def forward_next(self, batch, module):
        cur_input_key = module.input_key
        cur_target_key = module.target_key

        for m in module.children():
            if hasattr(m, "input_key") and len(m._modules) != 0 and not m.is_lower_trackable:
                batch = self.forward_next(batch=batch, module=m)
                cur_target_key = m.target_key
            elif hasattr(m, "input_key"):
                inputs = [batch[m.input_key]] if isinstance(m.input_key, str) else [batch[ikey] for ikey in m.input_key]
                batch[m.target_key] = m(*inputs)
                cur_target_key = m.target_key
            else:
                inputs = (
                    [batch[cur_input_key]]
                    if isinstance(cur_input_key, str)
                    else [batch[ikey] for ikey in cur_input_key]
                )
                batch[cur_target_key] = m(*inputs)
            cur_input_key = cur_target_key

            del_keys = getattr(m, "del_keys", {})
            del_keys = dict.fromkeys(del_keys)
            batch = {key: value for key, value in batch.items() if key not in del_keys}

        del_keys = getattr(module, "del_keys", {})
        del_keys = dict.fromkeys(del_keys)

        batch = {key: value for key, value in batch.items() if key not in del_keys}

        return batch

    def init_weights(self):
        def _init_weights(module, root=True):
            if not root and hasattr(module, "init_weights"):
                module.init_weights()
            else:
                root = False
                for m in module.children():
                    _init_weights(m, root=False)

        _init_weights(self)

from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from mmcv.utils import Registry, build_from_cfg


def recursive_build(
    cfg: dict | list | tuple,
    registry: Registry,
    module_name: Optional[str] = None,
    init_cfg: dict | None = None,
    input_key: str = "inputs",
    target_key: str = "logits",
):
    if isinstance(cfg, (list, tuple)):
        cfg_dict = {f"{name}": sub_cfg for name, sub_cfg in enumerate(cfg)}
    elif isinstance(cfg, dict):
        cfg_dict = cfg
    else:
        raise ValueError(
            "Config should be dict, list or tuple, but got {type(cfg)}"
        )

    init_cfg = cfg_dict.pop("init_cfg", init_cfg)

    input_key = cfg_dict.pop("input_key", input_key)
    target_key = cfg_dict.pop("target_key", target_key)
    module_name = cfg_dict.pop("module_name", module_name)

    if "type" in cfg_dict:
        module = build_from_cfg(cfg=cfg_dict, registry=registry)
    else:
        module = OrderedDict()
        for idx, (name, sub_cfg) in enumerate(cfg_dict.items()):
            sub_module_name, sub_module = recursive_build(
                cfg=sub_cfg,
                module_name=f"{name}",
                registry=registry,
                init_cfg=init_cfg,
                input_key=input_key if idx == 0 else target_key,
                target_key=target_key,
            )
            module[sub_module_name] = sub_module
        module = build_from_cfg(
            cfg=dict(
                type="Pipeline",
                input_key=input_key,
                target_key=target_key,
                builded_modules=module,
            ),
            registry=registry,
        )

    if module_name is None:
        return module
    return (module_name, module)


build = recursive_build
build_model = build
build_pipeline = build

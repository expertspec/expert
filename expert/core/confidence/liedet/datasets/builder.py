from mmcv.utils import Registry
from torch.utils.data import Dataset


datasets = Registry("datasets")


def build_dataset(cfg: dict) -> Dataset:
    return datasets.build(cfg)

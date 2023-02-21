from torch.utils.data import Dataset

from mmcv.utils import Registry

datasets = Registry("datasets")


def build_dataset(cfg: dict) -> Dataset:
    return datasets.build(cfg)

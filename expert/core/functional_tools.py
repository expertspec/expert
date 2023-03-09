from __future__ import annotations
import torch
from torch import Tensor
import torchvision.transforms as transforms
from typing import Tuple
import numpy as np
import gdown
import cv2
import os



def get_torch_home() -> str:
    """Get Torch Hub cache directory used for storing downloaded models and weights."""

    torch_home = os.path.expanduser(
        os.getenv(
            "TORCH_HOME",
            os.path.join(os.getenv("DG_CACHE_HOME", "~/.cache"), "torch")
        )
    )

    return torch_home


def get_model_folder(model_name: str, url: str) -> str:
    """Load folder with model weights from remote storage."""

    torch_home = get_torch_home()
    model_dir = os.path.join(torch_home, "checkpoints")
    os.makedirs(model_dir, exist_ok=True)

    cached_dir = os.path.join(model_dir, model_name)
    if not os.path.exists(cached_dir):
        gdown.download_folder(url, output=cached_dir, quiet=False)

    return cached_dir


def get_model_weights(model_name: str, url: str) -> str:
    """Load model weights from remote storage."""

    model_dir = os.path.join(get_torch_home(), "checkpoints")
    os.makedirs(model_dir, exist_ok=True)

    cached_dir = os.path.join(model_dir, model_name)
    if not os.path.exists(cached_dir):
        gdown.download(url, output=cached_dir, quiet=False)

    return cached_dir


class Rescale:
    """Rescale image to a given size."""

    def __init__(self, output_size: Tuple | int) -> None:
        """
        Args:
            output_size (Tuple | int): Desired output size.
        """

        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if isinstance(self.output_size, int):
            out_height, out_width = self.output_size, self.output_size
        else:
            out_height, out_width = self.output_size

        out_height, out_width = int(out_height), int(out_width)
        image = cv2.resize(image, (out_height, out_width),
                           interpolation=cv2.INTER_AREA)

        return image


class ToTensor:
    """Convert ndarrays to tensors."""

    def __call__(self, image: np.ndarray) -> Tensor:
        # Swap color axis.
        image = np.transpose(image, axes=(2, 0, 1))

        return torch.from_numpy(image).float()


class Normalize:
    """Normalize tensor image."""

    def __call__(self, image: Tensor) -> Tensor:
        normalization = transforms.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        return normalization(image).float()

from __future__ import annotations

import os
from typing import Tuple

import cv2
import gdown
import numpy as np
import torch
import torchvision.transforms as transforms
from torch import Tensor

import hashlib
import shutil
import sys
import tempfile
from tqdm.auto import tqdm
from urllib.request import urlopen, Request


def get_torch_home() -> str:
    """Get Torch Hub cache directory used for storing downloaded models and weights."""

    torch_home = os.path.expanduser(
        os.getenv(
            "TORCH_HOME",
            os.path.join(os.getenv("DG_CACHE_HOME", "~/.cache"), "torch"),
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


def get_model_weights_url(model_name: str, url: str) -> str:
    """Load model weights from remote storage."""

    model_dir = os.path.join(get_torch_home(), "checkpoints")
    os.makedirs(model_dir, exist_ok=True)

    cached_dir = os.path.join(model_dir, model_name)
    if not os.path.exists(cached_dir):
        download_url_to_file(url, dst=cached_dir)

    return cached_dir


def download_url_to_file(url: str, dst: str, hash_prefix: str = None, progress: bool = True) -> str:
    """Download object at the given URL to a local path.
    
    Args:
        url (str): URL of the object to download.
        dst (str): Full path where object will be saved, e.g. `/tmp/temporary_file`.
        hash_prefix (str, optional): If not None, the SHA256 downloaded file should start with `hash_prefix`.
            Defaults to None.
        progress (bool, optional): whether or not to display a progress bar to stderr.
            Defaults to True.
    
    Example:
        >>> torch.hub.download_url_to_file('https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth', '/tmp/temporary_file')
    """
    file_size = None
    # We use a different API for python2 since urllib(2) doesn't recognize the CA
    # certificates in older Python
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, "getheaders"):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        if hash_prefix is not None:
            sha256 = hashlib.sha256()
        with tqdm(total=file_size, disable=not progress,
                  unit="B", unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                if hash_prefix is not None:
                    sha256.update(buffer)
                pbar.update(len(buffer))

        f.close()
        if hash_prefix is not None:
            digest = sha256.hexdigest()
            if digest[:len(hash_prefix)] != hash_prefix:
                raise RuntimeError('invalid hash value (expected "{}", got "{}")'
                                   .format(hash_prefix, digest))
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)
    
    return dst


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
        image = cv2.resize(
            image, (out_height, out_width), interpolation=cv2.INTER_AREA
        )

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
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

        return normalization(image).float()

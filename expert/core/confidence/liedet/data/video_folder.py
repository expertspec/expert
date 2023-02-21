from __future__ import annotations

import copy
from glob import glob
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms as V

from liedet.data.video_reader import VideoReader


class VideoFolder(Dataset):
    """Base Video Folder Dataset

    The class implements base logic to represent tree of folders with video files as dataset.
    It extracts target label of video file from its filename.

    Example:

    .. code-block:: python

        from torch.data.utils import DataLoader
        from torchvision import transforms as V
        from torchaudio import transforms as A

        from liedet.data import VideoFolder

        batch_video_transforms = V.Compose([
            V.RandomResizedCrop(240),
            V.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
            V.ColorJitter(0.8, 0.8, 0.8, 0.2),
        ])
        batch_audio_transforms = A.MelSpectrogram(sample_rate=16000, normalized=True)

        window_video_transforms = V.Compose([
            V.RandomCrop(224),
        ])
        window_audio_transforms = V.Compose([
            A.Resample(orig_freq=48000, new_freq=16000),
            A.Vol(2),
        ])


        dataset = VideoFolder(
            root="data/interviews",
            pattern=".mp4",
            window=30*10,
            video_fps=30,
            audio_fps=48000,
            label_key="label",
            video_transforms=batch_video_transforms,
            audio_transforms=batch_audio_transforms,
            video_per_window_transforms=window_video_transforms,
            audio_per_window_transforms=window_audio_transforms,
            height=320,
            width=480,
        )

        for window in dataset:
            video_frames, audio_frames, label = (
                window["video_frames"],
                window["audio_frames"],
                window["label"],
            )

        train_set, valid_set = dataset.split(valid_size=0.2, by_file=True, balanced=True)

        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

        for batch in train_loader:
            video_frames, audio_frames, labels = (
                batch["video_frames"],
                batch["audio_frames"],
                batch["labels"],
            )
    """

    def __init__(
        self,
        root: str,
        pattern: str = ".mp4",
        window: int = 1,
        video_fps: int = 30,
        audio_fps: int = 48000,
        label_key: str = "label",
        video_transforms: V.Compose | None = None,
        audio_transforms: V.Compose | None = None,
        video_per_window_transforms: V.Compose | None = None,
        audio_per_window_transforms: V.Compose | None = None,
        height: int | None = None,
        width: int | None = None,
        mono: bool = True,
        bridge: str = "torch",
        valid_size: float | None = None,
        split_by_file: bool = True,
        balanced_valid_set: bool = True,
        seed: int | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            root (str): path to dataset root folder
            pattern (str, optional): filename pattern to recursive search files in root folder.
                Defaults to ".mp4".
            window (int, optional): number of video frames per window.
                Defaults to 1.
            video_fps (int, optional): target number of video frames per second.
                Source video fps is adjusted to this value.
                Defaults to 30.
            audio_fps (int, optional): target number of audio signals per second (Hz).
                Source audio fps is adjusted to this value.
                Defaults to 48000.
            label_key (str, optional): label target key in dict which is returned by get_meta method.
                Defaults to "label".
            video_transforms (V.Compose | None, optional): composition of transforms as preprocessing pipeline
                for video frames which applies to batch of window of video frames.
                Defaults to None.
            audio_transforms (V.Compose | None, optional): composition of transforms as preprocessing pipeline
                for audio frames which applies to batch of windowses of audio frames.
                Defaults to None.
            video_per_window_transforms (V.Compose | None, optional): composition of transforms
                as preprocessing pipeline for video frames which applies to each window of video frames one-by-one.
                Defaults to None.
            audio_per_window_transforms (V.Compose | None, optional): composition of transforms
                as preprocessing pipeline for audio frames which applied to each window of audio frames one-by-one.
                Defaults to None.
            height (int | None, optional): target height size of video frame.
                Source height of video frame is adjusted to this value.
                Defaults to None.
            width (int | None, optional): target width size of video frame.
                Source width of video frame is adjusted to this value.
                Defaults to None.
            mono (bool, optional): boolean flag to use single mono channel, otherwise two stereo channels is used.
                Defaults to True.
            bridge (str, optional): type of tensors returned by decord AVReader.
                Defaults to "torch".
            valid_size (float | None, optional): size of validation part of dataset.
                If passed then dataset will be splitted during initialization.
                Defaults to None.
            split_by_file (bool, optional): boolean flag to split dataset into subsets
                where single video file can be used only in one subset,
                otherwise windowses from single video file can be used in multiple subsets.
                Defaults to True.
            balanced_valid_set (bool, optional): boolean flag to balance subsets with respect to target label,
                otherwise generates unbalanced subsets.
                Defaults to True.
            seed (int | None, optional): initial state of random numbers generator.
                Defaults to None.
        """
        self.root = root
        self.pattern = pattern

        self.files = self.get_files()
        self.metas = self.get_metas()
        self.label_key = label_key
        self.labels = self.get_labels()
        self.per_file_frames_num = self.get_per_files_frames_num()

        self.window = window
        self.video_fps = video_fps
        self.audio_fps = audio_fps

        self.order = self.get_order()

        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

        self.video_transforms = video_transforms
        self.audio_transforms = audio_transforms
        self.video_per_window_transforms = video_per_window_transforms
        self.audio_per_window_transforms = audio_per_window_transforms

        self.bridge = bridge
        self.mono = mono

        self.height = height
        self.width = width

        self.valid_size: float | None = valid_size
        self._train_set: VideoFolder | None = None
        self._valid_set: VideoFolder | None = None
        self.split_by_file = split_by_file
        self.balanced_valid_set = balanced_valid_set
        if self.valid_size is not None:
            self._train_set, self._valid_set = self.split(
                valid_size=self.valid_size, by_file=split_by_file, balanced_valid_set=balanced_valid_set
            )

    def get_files(self) -> list[str]:
        """Recursively traverse directories in the root folder and returns a list of files that match the pattern.

        Returns:
            list[str]: list of video files.
        """
        pathname = f"{self.root}/**/*{self.pattern}"
        files = sorted(glob(pathname=pathname, recursive=True))

        return list(files)

    def get_metas(self) -> list[dict[str, Any]]:
        """Extracting meta information from filenames.

        Returns:
            list[dict[str, Any]]: list of dictionaries with appropriate files' meta information.
        """
        return [self.get_meta(path=path) for path in self.files]

    def get_meta(self, path: str) -> dict[str, str]:
        """Base function for extracting meta information from filename.

        Args:
            path (str): path to file.

        Returns:
            dict[str, str]: base dictionary with label only.
        """
        return dict(label=Path(path).name.split("_")[-1].split(".")[0])

    def get_labels(self) -> Tensor:
        """Extracting labels from files' meta information.

        Returns:
            Tensor: tensor of target labels.
        """
        return torch.tensor([int(meta[self.label_key]) for meta in self.metas], dtype=torch.long)

    def get_order(self) -> Tensor:
        """Building order of windowses depending on window size and number of video file frames.

        Returns:
            Tensor: tensor of pairs (file index, start frame).
        """
        return torch.tensor(
            [
                (i, j * self.window)
                for i, num_frames in enumerate(self.per_file_frames_num)
                for j in range(int(num_frames // self.window) - 1)
            ]
        )

    def get_per_files_frames_num(self) -> list[int]:
        """Extracting number of video frames from each video file.

        Returns:
            list[int]: list of number of video frame for each video file.
        """
        per_file_frames_num = []
        for file in self.files:
            cap = cv2.VideoCapture(file)
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            per_file_frames_num.append(n_frames)
            cap.release()

        return per_file_frames_num

    def split(
        self,
        valid_size: float = 0.2,
        by_file: bool = True,
        balanced_valid_set: bool = True,
        seed: int | None = None,
    ) -> tuple[VideoFolder, VideoFolder]:
        """Splitting dataset to training and validation subsets.

        Args:
            valid_size (float, optional): size of validation subset.
                Defaults to 0.2.
            by_file (bool, optional): boolean flag to split dataset into subsets
                where single video file can be used only in one subset,
                otherwise windowses from single video file can be used in multiple subsets.
                Defaults to True.
            balanced_valid_set (bool, optional): boolean flag to balance subsets with respect to target label,
                otherwise generates unbalanced subsets.
                Defaults to True.
            seed (int | None, optional): initial state of random numbers generator.
                Defaults to None.

        Returns:
            tuple[VideoFolder, VideoFolder]: tuple of training and validation subsets.
        """
        _rng = self.rng
        if seed is not None:
            _rng = np.random.default_rng(seed=seed)

        train_set = copy.deepcopy(self)
        valid_set = copy.deepcopy(self)

        labels: Tensor
        if by_file:
            labels = self.labels
        else:
            labels = self.labels[self.order[..., 0]]

        total_samples = self.order.size(0)

        if balanced_valid_set:
            counts: Tensor
            unique_labels, counts = labels.unique(return_counts=True)
            min_size = counts.min()
            valid_samples = int(min_size * valid_size)

            train_indices: list[Tensor] | Tensor = []
            valid_indices: list[Tensor] | Tensor = []
            for label in unique_labels:
                indices = torch.argwhere(labels == label).T[0]
                samples = indices.size(0)

                _indices = np.arange(samples)
                _rng.shuffle(_indices)

                valid_indices.extend(indices[_indices[:valid_samples]])  # type: ignore
                train_indices.extend(indices[_indices[valid_samples:]])  # type: ignore
        else:
            valid_samples = int(total_samples * valid_size)

            indices = np.arange(total_samples)
            _rng.shuffle(indices)

            valid_indices = indices[:valid_samples]
            train_indices = indices[valid_samples:]

        train_indices = torch.tensor(train_indices, dtype=torch.long)
        valid_indices = torch.tensor(valid_indices, dtype=torch.long)
        if by_file:
            train_indices = torch.isin(train_set.order[..., 0], train_indices)
            valid_indices = torch.isin(valid_set.order[..., 0], valid_indices)
            train_set.order = train_set.order[train_indices]
            valid_set.order = valid_set.order[valid_indices]
        else:
            train_set.order = train_set.order[train_indices]
            valid_set.order = valid_set.order[valid_indices]

        self._train_set, self._valid_set = train_set, valid_set

        return self._train_set, self._valid_set

    @property
    def train_set(self) -> VideoFolder:
        """
        Returns:
            VideoFolder: training subset.
        """
        if self._train_set is not None:
            return self._train_set

        self._train_set, self._valid_set = self.split()

        return self._train_set

    @property
    def valid_set(self) -> VideoFolder:
        """
        Returns:
            VideoFolder: validation subset.
        """
        if self._valid_set is not None:
            return self._valid_set

        self._train_set, self._valid_set = self.split()

        return self._valid_set

    def __len__(self) -> int:
        """
        Returns:
            int: number of windowses over all video files.
        """
        return self.order.size(0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Gets window of the dataset.

        Args:
            idx (int): index of windows in order.

        Returns:
            dict[str, Any]: dictionary with `video_frames`, `audio_frames`, `labels` and `meta` keys.
        """
        file_idx, start_frame_idx = self.order[idx, :]

        video_reader = VideoReader(
            uri=self.files[file_idx],
            video_fps=self.video_fps,
            audio_fps=self.audio_fps,
            video_transforms=self.video_per_window_transforms,
            audio_transforms=self.audio_per_window_transforms,
            mono=self.mono,
            bridge=self.bridge,
            height=self.height,
            width=self.width,
        )

        frames = video_reader[start_frame_idx : start_frame_idx + self.window]

        if self.video_transforms is not None:
            frames["video_frames"] = self.video_transforms(frames["video_frames"])
        if self.audio_transforms is not None:
            frames["audio_frames"] = self.audio_transforms(frames["audio_frames"])

        label = self.labels[file_idx]

        return {"labels": label, **frames}

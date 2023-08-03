from __future__ import annotations

import decord
import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.functional import resample
from torchvision import io
from torchvision import transforms as V


class AVReader(decord.AVReader):
    """Wrapper around decord.AVReader

    This class expands allowed types of indexing and allows to pass tensor as index.

    See also `decord.AVReader`_.

    .. _`decord.AVReader`: https://github.com/dmlc/decord/blob/master/python/decord/av_reader.py
    """

    def __getitem__(
        self, idx: int | slice | Tensor
    ) -> tuple[tuple[Tensor], Tensor]:
        """Gets video and audio frames with passed index(es).

        Args:
            idx (int | slice | Tensor): index(es) of video frames.

        Returns:
            tuple[tuple[Tensor], Tensor]: tuple of audio and video frames.
            Audio frames are returned as a tuple of Tensors because of different sizes.
            Video frames are returned as a Tensor.
        """
        assert (
            self.__video_reader is not None and self.__audio_reader is not None
        )

        if isinstance(idx, (slice, int)):
            return super().__getitem__(idx)

        return self.get_batch(idx)


class VideoReader(Dataset):
    """Base Video Reader Dataset

    The class implements base logic to extract video and audio frames (or windows of them) from video file.

    Example:

    .. code-block:: python

        from torchvision import transforms as V
        from torchaudio import transforms as A

        from liedet.data import VideoReader

        video_transforms = V.Compose([
            V.Resize(224),
            V.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ])
        audio_transforms = V.Compose([
            A.Resample(orig_freq=48000, new_freq=16000),
            A.MelSpectrogram(sample_rate=16000, normalized=True),
        ])

        video_reader = VideoReader(
            uri="assets/example.mp4",
            mono=True,
            height=320,
            width=480,
            video_fps=30,
            audio_fps=48000,
            video_transforms=video_transforms,
            audio_transforms=audio_transforms,
        )

        # one-by-one
        for frame in video_reader:
            video_frame, audio_frame = frame["video_frames"], frame["audio_frames"]

        window = video_fps * window_secs
        
        for start_index in range(0, len(video_reader), window):
            frames = video_reader[start_index : start_index + window]
            video_frames, audio_frames = frame["video_frames"], frame["audio_frames"]

    """

    def __init__(
        self,
        uri: str,
        mono: bool = True,
        bridge: str = "torch",
        height: int | None = None,
        width: int | None = None,
        video_fps: int | None = None,
        audio_fps: int | None = None,
        video_transforms: V.Compose | None = None,
        audio_transforms: V.Compose | None = None,
        **kwargs,
    ) -> None:
        """
        Args:
            uri (str): path to video file.
            mono (bool, optional): boolean flag to use single mono channel, otherwise two stereo channels is used.
                Defaults to True.
            bridge (str, optional): type of tensors returned by decord AVReader.
                Defaults to "torch".
            height (int | None, optional): target height size of video frame.
                Source height of video frame is adjusted to this value.
                Defaults to None.
            width (int | None, optional): target width size of video frame.
                Source width of video frame is adjusted to this value.
                Defaults to None.
            video_fps (int, optional): target number of video frames per second.
                Source video fps is adjusted to this value.
                Defaults to 30.
            audio_fps (int, optional): target number of audio signals per second (Hz).
                Source audio fps is adjusted to this value.
                Defaults to 48000.
            video_transforms (V.Compose | None, optional): composition of transforms
                as preprocessing pipeline for video frames which applies to each frame/window of video frames one-by-one.
                Defaults to None.
            audio_transforms (V.Compose | None, optional): composition of transforms
                as preprocessing pipeline for audio frames which applied to each frame/window of audio frames one-by-one.
                Defaults to None.
        """
        decord.bridge.set_bridge(new_bridge=bridge)

        self.uri = uri

        video_frames, audio_frames, meta = io.read_video(
            filename=uri, start_pts=0, end_pts=1
        )
        (
            self.orig_height,
            self.orig_width,
            self.num_video_channels,
        ) = video_frames.size()[1:]
        self.height = height if height is not None else self.orig_height
        self.width = width if width is not None else self.orig_width
        self.num_audio_channels = audio_frames.size()[0]

        self.orig_video_fps = meta["video_fps"]
        try:
            self.orig_audio_fps = meta["audio_fps"]
        except KeyError:
            self.orig_audio_fps = 0
        self.video_fps = (
            video_fps if video_fps is not None else self.orig_video_fps
        )
        self.audio_fps = (
            audio_fps if audio_fps is not None else self.orig_audio_fps
        )

        self.avreader = AVReader(
            uri=uri,
            sample_rate=self.orig_audio_fps,
            mono=mono,
            num_threads=1,
            height=self.height,
            width=self.width,
        )
        self.num_video_frames = len(self.avreader)
        self.num_audio_frames = (
            self.avreader._AVReader__audio_reader._num_samples_per_channel
        )

        self.meta = dict(
            height=self.height,
            width=self.width,
            orig_height=self.orig_height,
            orig_width=self.orig_width,
            num_video_channels=self.num_video_channels,
            num_audio_channels=self.num_audio_channels,
            video_fps=self.video_fps,
            audio_fps=self.audio_fps,
            orig_video_fps=self.orig_video_fps,
            orig_audio_fps=self.orig_audio_fps,
            num_video_frames=self.num_video_frames,
            num_audio_frames=self.num_audio_frames,
        )

        self.video_transforms = video_transforms
        self.audio_transforms = audio_transforms

    def __len__(self) -> int:
        """
        Returns:
            int: number of video frames.
        """
        return len(self.avreader)

    def _validate_indices(
        self, idx: slice, length: int
    ) -> tuple[int, int, int]:
        start, stop, step = idx.start, idx.stop, idx.step
        if start is None:
            start = 0
        elif start < 0:
            start += length
        if stop is None:
            stop = length
        elif stop < 0:
            stop += length
        if step is None:
            step = 1

        if start < 0:
            raise IndexError(f"Invalid start index: {idx.start}")
        if stop < 0:
            raise IndexError(f"Invalid stop index: {idx.stop}")

        return start, stop, step

    def _to_real_indices(
        self,
        start: int,
        stop: int,
        step: int,
        orig_freq: int,
        expected_freq: int,
        length: int,
    ) -> tuple[Tensor, int]:
        freq_div = expected_freq / orig_freq
        real_step = step / freq_div
        real_start = start * real_step
        real_stop = real_start + (stop - start) * real_step

        if real_stop > length:
            real_stop = length
            real_start = length - (stop - start) * real_step

        indices = torch.arange(real_start, real_stop, real_step).floor().long()

        pad_size = indices.size(0)
        indices = indices[indices >= 0]
        pad_size -= indices.size(0)

        return indices, pad_size

    def _pad(
        self, x: Tensor, pad_size: int, dim: int = 0, start: bool = False
    ) -> Tensor:
        if pad_size == 0:
            return x

        h = x.moveaxis(dim, 0)
        if start:
            # FIXME: It will be better to use .zeros and clip gradient,
            #   but catalyst's BackwardCallback with grad_clip_fn has bag: it uses undefined variable `model`
            h = torch.cat((torch.rand(pad_size, *h.size()[1:]), h), dim=0)
        else:
            # FIXME: It will be better to use .zeros and clip gradient,
            #   but catalyst's BackwardCallback with grad_clip_fn has bag: it uses undefined variable `model`
            h = torch.cat((h, torch.rand(pad_size, *h.size()[1:])), dim=0)
        h = h.moveaxis(0, dim)

        return h

    def __getitem__(self, idx: int | slice | Tensor) -> dict:
        """Gets single frame, several frames or window of the video file.

        Args:
            idx (int | slice | Tensor): index(es) or slice of indexes to extract from video file.

        Returns:
            dict[str, Any]: dictionary with `video_frames`, `audio_frames` and `meta` keys.
        """
        if isinstance(idx, slice):
            start, stop, step = self._validate_indices(
                idx, self.num_video_frames
            )
            indices, pad_size = self._to_real_indices(
                start,
                stop,
                step,
                self.orig_video_fps,
                self.video_fps,
                len(self.avreader),
            )

            start, stop = indices.min(), indices.max()  # type: ignore
            audio_frames, video_frames = self.avreader[start : stop + 1]
            video_frames = rearrange(video_frames, "t h w c -> t c h w")
            video_frames = video_frames[indices - start].float()

            video_frames = self._pad(video_frames, pad_size=pad_size, dim=0)

            expected_audio_size = int(
                video_frames.size(0) * self.audio_fps / self.video_fps
            )
            if len(audio_frames) != 0:
                audio_frames = torch.cat(audio_frames, dim=-1).float()
                audio_frames = resample(
                    audio_frames, self.orig_audio_fps, self.audio_fps
                )
                audio_frames = self._pad(
                    audio_frames,
                    pad_size=expected_audio_size - audio_frames.size(-1),
                    dim=-1,
                    start=True,
                )
            else:
                # FIXME: It will be better to use .zeros and clip gradient,
                #   but catalyst's BackwardCallback with grad_clip_fn has bag: it uses undefined variable `model`
                audio_frames = torch.rand(
                    self.num_audio_channels, expected_audio_size
                )

            if self.video_transforms is not None:
                video_frames = self.video_transforms(video_frames)
            if self.audio_transforms is not None:
                audio_frames = self.audio_transforms(audio_frames)
        else:
            audio_frames, video_frames = self.avreader[idx]
            audio_frames = audio_frames.float()
            video_frames = video_frames.float()

            video_frames = rearrange(video_frames, "h w c -> c h w")

            expected_audio_size = int(self.audio_fps / self.video_fps)
            audio_frames = resample(
                audio_frames, self.orig_audio_fps, self.audio_fps
            )
            audio_frames = self._pad(
                audio_frames,
                pad_size=expected_audio_size - audio_frames.size(-1),
                dim=-1,
                start=True,
            )

            if self.video_transforms is not None:
                video_frames = self.video_transforms(video_frames)
            if self.audio_transforms is not None:
                audio_frames = self.audio_transforms(audio_frames)

        return {
            "video_frames": video_frames,
            "audio_frames": audio_frames,
            "meta": self.meta,
        }

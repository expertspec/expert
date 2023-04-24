from __future__ import annotations

from collections import OrderedDict
from os import PathLike
from typing import Tuple

import cv2
import numpy as np


class Cache:
    """Caching class for decoding videos.

    If the same video frame is cached and used a
    second time, there is no need to decode it twice.

    Args:
        capacity (int): Buffer size for storing frames.

    Raises:
        ValueError: If "capacity" is not a positive integer.
    """

    def __init__(self, capacity: int) -> None:
        self._cache = OrderedDict()
        self._capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("Capacity must be a positive integer.")

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return len(self._cache)

    def put(self, key: int, val: np.ndarray):
        if key in self._cache:
            return
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = val

    def get(self, key: int, default: int | None = None):
        val = self._cache[key] if key in self._cache else default

        return val


class VideoReader:
    """Class for decoding video to a list object.

    This video wrapper class decodes the video and provides access to frames.

    Args:
        filename (str | PathLike): Path to local video file.
        cache_capacity (int, optional): Buffer size for storing frames. Defaults to 10.

    Raises:
        IndexError: If the entered frame index is outside the allowed range of integer values.
        IndexError: If the entered frame index is out of range.
        StopIteration: If the end of the video has been reached.
    """

    def __init__(
        self, filename: str | PathLike, cache_capacity: int = 10
    ) -> None:
        self._vcap = cv2.VideoCapture(filename)
        self._cache = Cache(cache_capacity)
        self._position = 0

        self._width = int(self._vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._vcap.get(cv2.CAP_PROP_FPS)
        self._frame_cnt = int(self._vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._fourcc = self._vcap.get(cv2.CAP_PROP_FOURCC)

    @property
    def vcap(self) -> cv2.VideoCapture:
        """Get VideoCapture object.

        Returns:
            cv2.VideoCapture: Raw VideoCapture object.
        """
        return self._vcap

    @property
    def opened(self) -> bool:
        """Check whether the video is opened.

        Returns:
            bool: Indicate whether the video is opened.
        """
        return self._vcap.isOpened()

    @property
    def width(self) -> int:
        """Get width of video frames.

        Returns:
            int: Width of video frames.
        """
        return self._width

    @property
    def height(self) -> int:
        """Get height of video frames.

        Returns:
            int: Height of video frames.
        """
        return self._height

    @property
    def resolution(self) -> Tuple[int, int]:
        """Get Video resolution (width, height).

        Returns:
            Tuple: Video resolution (width, height).
        """
        return (self._width, self._height)

    @property
    def fps(self) -> float:
        """Get FPS of the video.

        Returns:
            float: FPS of the video.
        """
        return self._fps

    @property
    def frame_cnt(self) -> int:
        """Get total number frames.

        Returns:
            int: Total frames of the video.
        """
        return self._frame_cnt

    @property
    def fourcc(self) -> str:
        """Get four character code.

        Returns:
            str: "Four character code" of the video.
        """
        return self._fourcc

    @property
    def position(self) -> int:
        """Get current cursor position.

        Returns:
            int: Current cursor position, indicating frame decoded.
        """
        return self._position

    def _get_real_position(self) -> int:
        return int(round(self._vcap.get(cv2.CAP_PROP_POS_FRAMES)))

    def _set_real_position(self, frame_id) -> None:
        self._vcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        pos = self._get_real_position()
        for _ in range(frame_id - pos):
            self._vcap.read()
        self._position = frame_id

    def read(self) -> np.ndarray | None:
        """Read the next frame.

        If the next frame have been decoded before and in the cache, then
        return it directly, otherwise decode, cache and return it.

        Returns:
            ndarray or None: Returns the frame if successful, otherwise returns None.
        """
        if self._cache:
            image = self._cache.get(self._position)
            if image is not None:
                ret = True
            else:
                if self._position != self._get_real_position():
                    self._set_real_position(self._position)
                ret, image = self._vcap.read()
                if ret:
                    self._cache.put(self._position, image)
        else:
            ret, image = self._vcap.read()
        if ret:
            self._position += 1

        return image

    def get_frame(self, frame_id: int) -> np.ndarray | None:
        """Get frame by index.

        Args:
            frame_id (int): Index of the expected frame, 0-based.

        Returns:
            ndarray or None: Returns the frame if successful, otherwise returns None.
        """
        if frame_id < 0 or frame_id >= self._frame_cnt:
            raise IndexError(
                f'"frame_id" must be between 0 and {self._frame_cnt - 1}'
            )
        if frame_id == self._position:
            return self.read()
        if self._cache:
            image = self._cache.get(frame_id)
            if image is not None:
                self._position = frame_id + 1
                return image
        self._set_real_position(frame_id)
        ret, image = self._vcap.read()
        if ret:
            if self._cache:
                self._cache.put(self._position, image)
            self._position += 1
        return image

    def current_frame(self) -> np.ndarray | None:
        """Get the current frame (the frame is just visited).

        Returns:
            ndarray or None: If the video is fresh returns None, otherwise returns the frame.
        """
        if self._position == 0:
            return None
        return self._cache.get(self._position - 1)

    def __len__(self):
        return self.frame_cnt

    def __getitem__(self, index: int):
        if isinstance(index, slice):
            return [
                self.get_frame(i) for i in range(*index.indices(self.frame_cnt))
            ]

        if index < 0:
            index += self.frame_cnt
            if index < 0:
                raise IndexError("Index out of range.")

        return self.get_frame(index)

    def __iter__(self):
        self._set_real_position(0)

        return self

    def __next__(self):
        image = self.read()
        if image is not None:
            return image
        else:
            raise StopIteration

    next = __next__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vcap.release()

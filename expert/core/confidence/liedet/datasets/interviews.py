from __future__ import annotations

from pathlib import Path
from typing import Any

from liedet.data import VideoFolder
from liedet.datasets.builder import datasets


@datasets.register_module()
class Interviews(VideoFolder):
    """Video Folder implementation for Interviews Dataset."""

    def get_meta(self, path: str) -> dict[str, Any]:
        """Implementation of get_meta function for Interviews Dataset.

        Args:
            path (str): path to file.

        Returns:
            dict[str, int]: dictionary with `user`, `interview_id`, `question_id` and `label` keys.
        """
        filename = Path(path)
        user, interview_id = filename.parent.name.split("_")
        question_id, label = filename.name.split("_")
        label = label.split(".")[0]

        return dict(
            user=int(user),
            interview_id=int(interview_id),
            question_id=int(question_id),
            label=int(label),
        )

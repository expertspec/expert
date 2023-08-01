from __future__ import annotations

import torch
from einops import rearrange
from mediapipe.python.solutions import face_mesh
from torch import Tensor

from expert.core.confidence.liedet.models.base_module import BaseModule
from expert.core.confidence.liedet.models.detectors.landmarks.rotate_regressor import Regressor
from expert.core.confidence.liedet.models.registry import registry


@registry.register_module()
class FaceLandmarks(BaseModule):
    """Face Landmarks extractor.

    This models wraps Mediapipe Face Mesh for landmarks extraction
    and computes rotation angles to normalize face rotation.

    See also: `MediaPipe Face Mesh`_.

    .. _`MediaPipe Face Mesh`: https://google.github.io/mediapipe/solutions/face_mesh.html

    """

    def __init__(
        self,
        window: int | None = None,
        static_image_mode: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        normalize: bool = True,
        rotate: bool = True,
        init: bool = False,
        **kwargs,
    ):
        """
        Args:
            window (int | None, optional): number of frames per window.
                If None when static_image_mode is True by default,
                otherwise face mesh is tried to track faces over window frames.
                Defaults to None.
            static_image_mode (bool, optional): boolean flag to track face over frames.
                Defaults to True.
            min_detection_confidence (float, optional): minimun confidence threshold to select face.
                Defaults to 0.5.
            min_tracking_confidence (float, optional): minimun confidence threshold to track face.
                Defaults to 0.5.
            normalize (bool, optional): boolean flag to normalize landmarks.
                Defaults to True.
            rotate (bool, optional): boolean flag to normalize landmarks rotation
                and extract rotation angles.
                Defaults to True.
            init (bool, optional): boolean flag to initialize weights
                or load pretrained ones from config.
                Defaults to False.
        """
        super().__init__(**kwargs)

        self.window = window

        self.static_image_mode = static_image_mode if window is None else False
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.norm = normalize
        self.rot = rotate

        self.regressor = Regressor()

        if init:
            self.init_weights()

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Forwards input images to Mediapipe Face Mesh.

        Args:
            x (Tensor): batch or input images.

        Returns:
            Tensor: batch of landmarks [and rotation angles].
        """
        device = x.device

        self.regressor.to(device)
        self.regressor.eval()
        if self.window is not None:
            h = rearrange(x, "b t c h w -> b t h w c")
            # h = rearrange(tensor=x, pattern="(b t) c h w -> b t h w c", t=self.window)
        else:
            h = rearrange(tensor=x, pattern="(b t) c h w -> b t h w c", b=1)

        h = h.cpu().numpy().astype("uint8")
        is_face = []

        batch_landmarks = []
        for chunk in h:
            with face_mesh.FaceMesh(
                static_image_mode=self.static_image_mode,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            ) as fm:
                chunk_landmarks = []
                # FIXME: It will be better to use .zeros and clip gradient,
                #   but catalyst's BackwardCallback with grad_clip_fn has bag: it uses undefined variable `model`
                prev_landmarks = torch.rand(
                    size=(3, 478), dtype=torch.float
                ).to(device)
                for frame in chunk:
                    landmarks = fm.process(frame)
                    if landmarks.multi_face_landmarks:
                        landmarks = landmarks.multi_face_landmarks[0].landmark
                        landmarks = [
                            torch.tensor(
                                (landmark.x, landmark.y, landmark.z),
                                dtype=torch.float,
                            )
                            for landmark in landmarks
                        ]
                        landmarks = torch.stack(landmarks).T.to(device)
                        is_face.append(1)
                        prev_landmarks = landmarks
                    else:
                        landmarks = prev_landmarks
                        is_face.append(0)

                    chunk_landmarks.append(landmarks)
                batch_landmarks.append(torch.stack(chunk_landmarks))

        h = torch.stack(batch_landmarks)

        del batch_landmarks

        if self.norm:
            h = self.normalize(x=h)
        if self.rot:
            h = self.rotate(x=h)

        return h, is_face

    def normalize(self, x: Tensor) -> Tensor:
        """Min/Max normalization of landmarks.

        Args:
            x (Tensor): batch of input landmarks.

        Returns:
            Tensor: batch of normalized landmarks.
        """
        min_value, max_value = (
            x.min(dim=-1, keepdim=True).values,
            x.max(dim=-1, keepdim=True).values,
        )

        if min_value.allclose(max_value):
            return torch.zeros(x.size(), dtype=torch.float).to(x.device)

        return (x - min_value).abs() / (max_value - min_value).abs()

    def _rotate(self, x: Tensor, axis: Tensor, angles: Tensor) -> Tensor:
        """Rotates landmarks over single axis.

        Args:
            x (Tensor): batch of input landmarks.
            axis (Tensor): basis of target axis.
            angles (Tensor): batch of rotation angles.

        Returns:
            Tensor: batch of rotated landmarks.
        """
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        dot_products = torch.inner(axis, x)
        dot_products = rearrange(dot_products, "c k m b t f -> b t f (c k m)")
        cross_products = torch.linalg.cross(axis, x, dim=-1)

        return (
            cos_angles * x
            + sin_angles * cross_products
            + (1 - cos_angles) * dot_products * axis
        )

    def rotate(self, x: Tensor) -> Tensor:
        """Regresses rotations angles and normalizes landmarks rotation angles.

        Args:
            x (Tensor): batch of input landmarks.

        Returns:
            Tensor: batch of normalized landmarks and rotation angles.
        """
        h: Tensor = x - x.mean(dim=1, keepdim=True)
        h = h / torch.norm(h, dim=-1, keepdim=True)

        angles = self.regressor(h.flatten(start_dim=2))
        angles[..., 0] = -angles[..., 0]
        angles[..., 1] = -angles[..., 1]

        h = rearrange(h, "b t c f -> b t f c")
        angles = rearrange(angles, "b t (c k f) -> b t f k c", f=1, k=1)

        basis = torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=torch.float,
        )
        basis = basis.view(1, 1, 1, 3, 3).to(x.device)

        h = self._rotate(h, basis[..., 2], angles[..., 2])
        h = self._rotate(h, basis[..., 1], angles[..., 1])
        h = self._rotate(h, basis[..., 0], angles[..., 0])

        h = torch.cat((h, angles[..., 0, :]), dim=-2)
        h = rearrange(h, "b t f c -> b t c f").flatten(start_dim=-2)

        return h

from __future__ import annotations

from typing import List, Optional, Union

import albumentations as A
import cv2
import mediapipe
import numpy as np
import torch
from albumentations.pytorch.transforms import ToTensorV2

from expert.data.detection.inception_resnet_v1 import InceptionResnetV1


class FaceDetector:
    """Face detection and embedding implementation.

    FaceDetector processes an BGR image and returns a list of the detected face embeddings and bounding boxes.

    Example:
        >>> face_detector = FaceDetector(model_selection=0, min_detection_confidence=0.9)
    """

    def __init__(
        self,
        model_selection: Optional[int] = 0,
        min_detection_confidence: Optional[float] = 0.75,
        max_num_faces: Optional[int] = 10,
        device: Optional[Union[torch.device, None]] = None,
    ) -> None:
        """
        Args:
            model_selection (Optional[int]): 0 or 1. 0 to select a short-range model that works
                best for faces within 2 meters from the camera, and 1 for a full-range
                model best for faces within 5 meters. Defaults to 0.
            min_detection_confidence (Optional[float]): Minimum confidence value ([0.0, 1.0]) for face
                detection to be considered successful. Defaults to 0.75.
            max_num_faces (Optional[int]): Maximum number of faces to detect. Defaults to 10.
            device (Optional[Union[torch.device, None]): Device type on local machine (GPU recommended). Defaults to None.
        """
        super().__init__()

        self.max_num_faces = max_num_faces
        face_detector = mediapipe.solutions.face_detection
        self.face_detector = face_detector.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

        # Initialize InceptionResnetV1 on GPU device if available.
        self._device = torch.device("cpu")
        if device is not None:
            self._device = device

        self.face_embedder = InceptionResnetV1(
            pretrained="vggface2", device=self._device
        ).eval()

        # Declare an augmentation pipeline.
        self.transform = A.Compose(
            [
                A.Resize(width=224, height=224),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
                ToTensorV2(),
            ]
        )

    @property
    def device(self) -> torch.device:
        """Check the device type.

        Returns:
            torch.device: Device type on local machine.
        """
        return self._device

    @torch.no_grad()
    def detect(self, image: np.ndarray) -> List:
        """
        Args:
            image (np.ndarray): RGB image represented as numpy ndarray.

        Returns:
            List: List with detected face locations.
        """

        face_array = []
        image_height, image_width = image.shape[:2]
        prediction = self.face_detector.process(image)

        if prediction.detections:
            for n, idx in zip(
                range(self.max_num_faces), range(len(prediction.detections))
            ):
                bounding_box = prediction.detections[
                    idx
                ].location_data.relative_bounding_box
                face_location = [
                    [
                        int(bounding_box.xmin * image_width),
                        int(bounding_box.ymin * image_height),
                    ],
                    [
                        int(bounding_box.width * image_width),
                        int(bounding_box.height * image_height),
                    ],
                ]

                if sum([sum(loc) for loc in face_location]) == sum(
                    [sum(map(abs, loc)) for loc in face_location]
                ):
                    face_array.append(face_location)

        return face_array

    @torch.no_grad()
    def embed(self, image: np.ndarray) -> List:
        """Cropping and embedding area where the face is located.

        Args:
            image (np.ndarray): BGR image represented as numpy ndarray.

        Returns:
            List: List with detected face locations and embeddings.
        """

        face_batch = []
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_array = self.detect(image=image)

        if face_array is not None:
            for face_location in face_array:
                face_image = image[
                    face_location[0][1] : face_location[0][1]
                    + face_location[1][1],
                    face_location[0][0] : face_location[0][0]
                    + face_location[1][0],
                ]

                transformed_face = self.transform(image=face_image)["image"]
                in_face = transformed_face.unsqueeze(0).to(self._device)
                face_emb = (
                    self.face_embedder(in_face)[0].detach().cpu().tolist()
                )

                face_batch.append([face_emb, face_location])

        return face_batch

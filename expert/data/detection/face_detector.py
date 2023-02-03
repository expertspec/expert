from __future__ import annotations

import torch
import numpy as np
from typing import List, Tuple
import torchvision.transforms as transforms
import mediapipe
import cv2

from expert.data.detection.inception_resnet_v1 import InceptionResnetV1


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
        image = cv2.resize(image, (out_height, out_width), interpolation=cv2.INTER_AREA)
        
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
        normalization = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        
        return normalization(image).float()


class FaceDetector:
    """Face detection and embedding implementation.
    
    FaceDetector processes an BGR image and returns a list of the detected face embeddings and bounding boxes.
    
    Example:
        >>> face_detector = FaceDetector(model_selection=0, min_detection_confidence=0.9)
    """
    
    def __init__(
        self,
        model_selection: int = 0,
        min_detection_confidence: float = 0.75,
        max_num_faces: int = 10,
        device: torch.device | None = None
    ) -> None:
        """
        Args:
            model_selection (int, optional): 0 or 1. 0 to select a short-range model that works
                best for faces within 2 meters from the camera, and 1 for a full-range
                model best for faces within 5 meters.
            min_detection_confidence (float, optional): Minimum confidence value ([0.0, 1.0]) for face
                detection to be considered successful.
            max_num_faces (int, optional): Maximum number of faces to detect.
            device (torch.device | None, optional): Object representing device type.
        """
        super().__init__()
        
        self.max_num_faces = max_num_faces
        face_detector = mediapipe.solutions.face_detection
        self.face_detector = face_detector.FaceDetection(model_selection=model_selection,
            min_detection_confidence=min_detection_confidence)
        
        # Initialize InceptionResnetV1 on GPU device if available.
        self._device = torch.device("cpu")
        if device is not None:
            self._device = device
        
        self.face_embedder = InceptionResnetV1(pretrained="vggface2", device=self._device).eval()
        
        # Declare an augmentation pipeline.
        self.transform = transforms.Compose([
            Rescale(output_size=(224, 224)),
            ToTensor(),
            Normalize(),
        ])
    
    @property
    def device(self) -> torch.device:
        """Check the device type."""
        return self._device
    
    def detect(self, image: np.ndarray) -> List:
        """
        Args:
            image (np.ndarray): RGB image represented as numpy ndarray.
        """
        
        face_array = []
        image_height, image_width = image.shape[:2]
        prediction = self.face_detector.process(image)
        
        if prediction.detections:
            for n, idx in zip(range(self.max_num_faces), range(len(prediction.detections))):
                bounding_box = prediction.detections[idx].location_data.relative_bounding_box
                face_location = [[int(bounding_box.xmin * image_width), int(bounding_box.ymin * image_height)],
                                 [int(bounding_box.width * image_width), int(bounding_box.height * image_height)]]
                
                if sum([sum(loc) for loc in face_location]) == sum([sum(map(abs, loc)) for loc in face_location]):
                    face_array.append(face_location)
        
        return face_array
    
    @torch.no_grad()
    def embed(self, image: np.ndarray) -> List:
        """Cropping and embedding area where the face is located
        
        Args:
            image (np.ndarray): BGR image represented as numpy ndarray.
        """
        
        face_batch = []
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_array = self.detect(image=image)
        
        if face_array is not None:
            for face_location in face_array:
                face_image = image[face_location[0][1]:face_location[0][1]+face_location[1][1],
                                   face_location[0][0]:face_location[0][0]+face_location[1][0]]
                
                transformed_face = self.transform(face_image)
                in_face = transformed_face.unsqueeze(0).to(self._device)
                face_emb = self.face_embedder(in_face)[0].detach().cpu().tolist()
                
                face_batch.append([face_emb, face_location])
        
        return face_batch
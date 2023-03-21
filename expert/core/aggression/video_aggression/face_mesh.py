from __future__ import annotations

import cv2
import mediapipe
import numpy as np


# fmt: off
HEAD_INDEXES = np.array([
    0, 7, 10, 13, 14, 17, 21, 33, 37, 39, 40,
    46, 52, 53, 54, 55, 58, 61, 63, 65, 66, 67,
    70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95,
    103, 105, 107, 109, 127, 132, 133, 136, 144, 145, 146,
    148, 149, 150, 152, 153, 154, 155, 157, 158, 159, 160,
    161, 162, 163, 172, 173, 176, 178, 181, 185, 191, 234,
    246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284,
    285, 288, 291, 293, 295, 296, 297, 300, 308, 310, 311,
    312, 314, 317, 318, 321, 323, 324, 332, 334, 336, 338,
    356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380,
    381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398,
    400, 402, 405, 409, 415, 454, 466
])

LOW_PART = np.array([
    0, 11, 12, 13, 14, 15, 16, 17, 18, 32, 37, 38, 39, 40, 41, 42, 43, 57, 58, 61,
    62, 72, 73, 74, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 95,
    96, 106, 135, 136, 138, 140, 146, 148, 149, 150, 152, 169, 170, 171, 172, 175, 176, 178, 179, 180,
    181, 182, 183, 184, 185, 186, 191, 192, 194, 199, 200, 201, 202, 204, 208, 210, 211, 212, 214, 215,
    262, 267, 268, 269, 270, 271, 272, 273, 287, 288, 291, 292, 302, 303, 304, 306, 307, 308, 310, 311,
    312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 324, 325, 335, 364, 365, 367, 369, 375, 377,
    378, 379, 394, 395, 396, 397, 400, 402, 403, 404, 405, 406, 407, 408, 409, 410, 415, 416, 418, 421,
    422, 424, 428, 430, 431, 432, 433, 434, 435, 436
])

UPP_PART = np.array([
    7, 8, 9, 10, 21, 27, 28, 29, 30, 33, 46, 52, 53, 54, 55, 56, 63, 65, 66, 67,
    68, 69, 70, 71, 103, 104, 105, 107, 108, 109, 113, 124, 130, 139, 151, 156, 157, 158, 159, 160,
    161, 162, 168, 173, 189, 190, 193, 221, 222, 223, 224, 225, 226, 246, 247, 251, 257, 258, 259, 260,
    276, 282, 283, 284, 285, 286, 293, 295, 296, 297, 298, 299, 300, 301, 332, 333, 334, 336, 337, 338,
    368, 383, 384, 385, 386, 387, 389, 413, 414, 417, 441, 442, 443, 444, 445, 468, 469, 470, 471, 475
])
# fmt: on


class FaceMesh(object):
    """MediaPipe Face Mesh implementation.

    MediaPipe Face Mesh processes an RGB image and returns 478 face landmarks on each detected face.

    Args:
        static_image_mode (int, optional): Whether to treat the input images as a batch
            of static and possibly unrelated images or a video stream. Defaults to False.
        max_num_faces (int, optional): Maximum number of faces to detect. Defaults to 1.
        refine_landmarks (bool, optional): Whether to further refine the landmark coordinates
            around the eyes, lips and output additional landmarks around the irises. Defaults to True.
        min_detection_confidence (float, optional): Minimum confidence value ([0.0, 1.0]) for face
            detection to be considered successful. Defaults to 0.5.
        min_tracking_confidence (float, optional): Minimum confidence value ([0.0, 1.0]) for the
            face landmarks to be considered tracked successfully. Defaults to 0.5.

    Example:
        >>> import cv2
        >>> face_mesh = FaceMesh(static_image_mode=True, max_num_faces=1)
        >>> image = cv2.imread('test.jpg')
        >>> face_array = face_mesh.detect(image)
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        mesh_detector = mediapipe.solutions.face_mesh
        self.model = mesh_detector.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Args:
            image (np.ndarray): BGR image represented as numpy ndarray.
            normalize (bool, optional): Apply minimax normalization to facial landmarks.
        """

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_array = []
        prediction = self.model.process(image)

        if prediction.multi_face_landmarks:
            for idx in range(len(prediction.multi_face_landmarks)):
                landmarks = prediction.multi_face_landmarks[idx].landmark
                face_vector = np.array(
                    [
                        [landmark.x, landmark.y, landmark.z]
                        for landmark in landmarks
                    ]
                )

                if normalize:
                    # Normalization and centering of the face from -0.5 to 0.5 in width, height and depth of the face.
                    min_value = face_vector.min(axis=0)
                    max_value = face_vector.max(axis=0)

                    face_array = np.absolute(
                        face_vector - min_value
                    ) / np.absolute(max_value - min_value) - face_vector.mean(
                        axis=0
                    )

        return np.array(face_array)


def preprocess_face_vector(
    face_vector: np.ndarray, head_indexes: np.ndarray = HEAD_INDEXES
) -> np.ndarray:
    """Face landmarks preprocessing for head rotation prediction.

    Args:
        face_vector (np.ndarray): Face landmarks represented as numpy ndarray.
        head_indexes (np.ndarray): To determine angles of head rotation, landmarks of the face,
            eyes and lips are used. This allows to reduce size of the input vector.
    """

    # Leveling of face relief.
    head_vector = [face_vector[index] for index in head_indexes]
    vector_norm = np.linalg.norm(head_vector, axis=-1)
    head_vector = head_vector / np.tile(np.expand_dims(vector_norm, 1), [1, 3])

    # Linear straightening of head vector.
    head_vector = np.column_stack(head_vector).flatten()

    return head_vector

import pytest

from expert.data.detection.face_detector import FaceDetector
from expert.data.detection.inception_resnet_v1 import InceptionResnetV1
import torch
import cv2


@pytest.fixture()
def detector():
    detector = FaceDetector(device="cpu", model_selection=0, min_detection_confidence=0.75, max_num_faces=10)
    
    return detector


@pytest.fixture()
def resnet():
    resnet = InceptionResnetV1(device="cpu").eval()
    
    return resnet


@pytest.fixture()
def image():
    image = cv2.imread("test/data/test_image.jpeg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


def test_detect(detector, image):
    pred = detector.detect(image=image)
    
    assert pred is not None
    assert len(pred) == 1
    assert len(pred[0]) == 2
    assert len(pred[0][0]) == 2
    assert len(pred[0][1]) == 2


def test_embed(detector, image):
    pred = detector.embed(image=image)
    
    assert pred is not None
    assert len(pred) == 1
    assert len(pred[0]) == 2
    assert len(pred[0][0]) == 512
    assert len(pred[0][1]) == 2
    assert len(pred[0][1][0]) == 2
    assert len(pred[0][1][1]) == 2


def test_inception_resnet(resnet):
    data = torch.rand(1, 3, 224, 224)
    pred = resnet(data)
    
    assert pred is not None
    assert len(pred) == 1
    assert len(pred[0]) == 512

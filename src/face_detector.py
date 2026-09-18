# SmartDrive - Face Detector

import os
import urllib.request
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "/content/SmartDrive/models/blaze_face_short_range.tflite"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float16/latest/"
    "blaze_face_short_range.tflite"
)


def download_face_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(
            MODEL_URL,
            MODEL_PATH
        )


def create_face_detector():
    download_face_model()

    base_options = python.BaseOptions(
        model_asset_path=MODEL_PATH
    )

    options = vision.FaceDetectorOptions(
        base_options=base_options,
        min_detection_confidence=0.2
    )

    detector = vision.FaceDetector.create_from_options(
        options
    )

    return detector

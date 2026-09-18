# SmartDrive - Eye State Classifier

import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODEL_PATH = "../models/smartdrive_eye_model.keras"

model = load_model(MODEL_PATH)


def predict_eye(eye_image):
    gray = cv2.cvtColor(eye_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))

    image = gray.astype("float32") / 255.0
    image = np.expand_dims(image, axis=(0, -1))

    probability = float(model.predict(image, verbose=0)[0][0])

    if probability >= 0.5:
        state = "OPEN"
        confidence = probability
    else:
        state = "CLOSED"
        confidence = 1.0 - probability

    return state, confidence

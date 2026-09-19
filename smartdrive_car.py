"""
SmartDrive - Standalone real-time driver drowsiness prototype.
Place this file beside:
  smartdrive_eye_model.keras
  blaze_face_short_range.tflite

Install:
  pip install tensorflow opencv-python mediapipe numpy pyttsx3

Run:
  python smartdrive_car.py

Press Q to quit.
"""

import os
import time
import csv
import threading
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


MODEL_PATH = "smartdrive_eye_model.keras"
FACE_MODEL_PATH = "blaze_face_short_range.tflite"
LOG_FILE = "smartdrive_session_logs.csv"

CAMERA_INDEX = 0
DROWSY_SECONDS = 2.0
CRITICAL_SECONDS = 4.0


def speak(message):
    if pyttsx3 is None:
        return

    def worker():
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 165)
            engine.say(message)
            engine.runAndWait()
            engine.stop()
        except Exception:
            pass

    threading.Thread(target=worker, daemon=True).start()


class SmartDrive:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(MODEL_PATH)

        if not os.path.exists(FACE_MODEL_PATH):
            raise FileNotFoundError(FACE_MODEL_PATH)

        print("Loading eye CNN...")
        self.model = load_model(MODEL_PATH)

        print("Loading face detector...")
        base_options = python.BaseOptions(
            model_asset_path=FACE_MODEL_PATH
        )
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.2
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)

        self.blink_count = 0
        self.previous_closed = False
        self.closed_start = None
        self.maximum_closed = 0.0
        self.last_voice_time = 0.0

    def predict_eye(self, eye):
        if eye is None or eye.size == 0:
            return "UNKNOWN", 0.0

        gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))
        image = gray.astype("float32") / 255.0
        image = np.expand_dims(image, axis=(0, -1))

        p = float(self.model.predict(image, verbose=0)[0][0])

        if p >= 0.5:
            return "OPEN", p

        return "CLOSED", 1.0 - p

    def get_eye_crops(self, frame, detection):
        h, w = frame.shape[:2]
        keypoints = getattr(detection, "keypoints", None)

        if keypoints is None or len(keypoints) < 2:
            return None, None

        p1 = keypoints[0]
        p2 = keypoints[1]

        cx1 = int(p1.x * w)
        cy1 = int(p1.y * h)
        cx2 = int(p2.x * w)
        cy2 = int(p2.y * h)

        size = max(int(detection.bounding_box.height * 0.22), 35)

        def crop(cx, cy):
            half = size // 2
            xa = max(0, cx - half)
            ya = max(0, cy - half)
            xb = min(w, cx + half)
            yb = min(h, cy + half)
            return frame[ya:yb, xa:xb]

        return crop(cx1, cy1), crop(cx2, cy2)

    def get_status(self, both_closed):
        if both_closed:
            if self.closed_start is None:
                self.closed_start = time.time()
            current = time.time() - self.closed_start
            self.maximum_closed = max(self.maximum_closed, current)
        else:
            current = 0.0
            self.closed_start = None

        if current >= CRITICAL_SECONDS:
            return "CRITICAL", current
        if current >= DROWSY_SECONDS:
            return "DROWSY", current
        if current >= 1.0:
            return "WARNING", current
        return "SAFE", current

    def alert(self, status):
        if status == "SAFE":
            return

        now = time.time()
        if now - self.last_voice_time < 3.0:
            return

        self.last_voice_time = now

        if status == "WARNING":
            speak("Warning. Please stay alert.")
        elif status == "DROWSY":
            speak("Drowsiness detected. Please stay alert.")
        elif status == "CRITICAL":
            speak("Critical warning. Please stop the vehicle safely.")

    def save_log(self, final_status):
        exists = os.path.exists(LOG_FILE)

        with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            if not exists:
                writer.writerow([
                    "Session Time",
                    "Total Blinks",
                    "Maximum Eye Closure (sec)",
                    "Final Status"
                ])

            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.blink_count,
                round(self.maximum_closed, 2),
                final_status
            ])

    def run(self):
        camera = cv2.VideoCapture(CAMERA_INDEX)

        if not camera.isOpened():
            print("ERROR: Camera could not be opened.")
            return

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("=" * 60)
        print("SMARTDRIVE CAR MODE STARTED")
        print("Press Q to quit.")
        print("=" * 60)

        final_status = "SAFE"

        try:
            while True:
                ok, frame = camera.read()

                if not ok:
                    print("Camera frame could not be read.")
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                result = self.face_detector.detect(mp_image)

                left_state = "UNKNOWN"
                right_state = "UNKNOWN"
                left_conf = 0.0
                right_conf = 0.0
                both_closed = False

                if result.detections:
                    detection = result.detections[0]
                    box = detection.bounding_box

                    x = int(box.origin_x)
                    y = int(box.origin_y)
                    bw = int(box.width)
                    bh = int(box.height)

                    cv2.rectangle(
                        frame,
                        (x, y),
                        (x + bw, y + bh),
                        (0, 255, 0),
                        2
                    )

                    eye1, eye2 = self.get_eye_crops(
                        frame, detection
                    )

                    if eye1 is not None and eye2 is not None:
                        left_state, left_conf = self.predict_eye(eye1)
                        right_state, right_conf = self.predict_eye(eye2)

                        both_closed = (
                            left_state == "CLOSED"
                            and right_state == "CLOSED"
                        )

                if self.previous_closed and not both_closed:
                    self.blink_count += 1

                self.previous_closed = both_closed

                status, closed_time = self.get_status(both_closed)
                final_status = status
                self.alert(status)

                cv2.putText(
                    frame, "SMARTDRIVE", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2
                )
                cv2.putText(
                    frame, f"LEFT: {left_state} {left_conf:.0%}",
                    (25, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2
                )
                cv2.putText(
                    frame, f"RIGHT: {right_state} {right_conf:.0%}",
                    (25, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2
                )
                cv2.putText(
                    frame, f"BLINKS: {self.blink_count}",
                    (25, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2
                )
                cv2.putText(
                    frame, f"EYE CLOSED: {closed_time:.1f}s",
                    (25, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 255), 2
                )
                cv2.putText(
                    frame, f"STATUS: {status}",
                    (25, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2
                )

                cv2.imshow(
                    "SmartDrive - Driver Drowsiness Detection",
                    frame
                )

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()
            self.save_log(final_status)

            print()
            print("SMARTDRIVE SESSION FINISHED")
            print("Total blinks:", self.blink_count)
            print(
                "Maximum eye closure:",
                f"{self.maximum_closed:.2f} sec"
            )
            print("Log saved:", LOG_FILE)


if __name__ == "__main__":
    try:
        SmartDrive().run()
    except Exception as error:
        print("SmartDrive could not start.")
        print("Error:", error)

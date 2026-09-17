# SmartDrive — Real-Time Driver Drowsiness Detection

SmartDrive is a computer vision and deep learning project designed to detect driver drowsiness by monitoring eye states in real time.

## 🚗 Project Overview

Driver fatigue can reduce attention and reaction time. SmartDrive analyzes the driver's eyes using a webcam and a CNN-based eye-state classifier to identify prolonged eye closure.

## ✨ Features

* Real-time face detection
* Eye-state classification
* Open / Closed eye prediction
* Blink counting
* Eye-closure duration tracking
* Drowsiness detection using a time threshold
* Audio alarm when drowsiness is detected
* Session logging
* Drowsiness analytics and visualization
* Trained CNN model

## 🔄 System Pipeline

```text
Webcam
   ↓
Face Detection
   ↓
Eye Region Extraction
   ↓
CNN Eye-State Classification
   ↓
Blink & Eye-Closure Analysis
   ↓
Drowsiness Detection
   ↓
Audio Alert + Dashboard + Logs
```

## 🧠 Eye-State Classes

| Class | Meaning |
| ----: | ------- |
|     0 | Closed  |
|     1 | Open    |

The CNN model processes eye images resized to **64 × 64 grayscale**.

## 📊 Dataset

The model was trained using the **MRL Eye Dataset**, which contains images of open and closed eyes captured under different conditions.

Dataset information:
http://mrl.cs.vsb.cz/data/eyedataset/mrlEyes_2018_01.zip

## 🛠️ Technologies

* Python
* TensorFlow / Keras
* OpenCV
* MediaPipe
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Google Colab

## 📁 Project Structure

```text
SmartDrive/
│
├── assets/
│
├── logs/
│   └── smartdrive_session_logs.csv
│
├── models/
│   └── smartdrive_eye_model.keras
│
├── notebooks/
│
├── src/
│   ├── drowsiness_detector.py
│   ├── eye_classifier.py
│   ├── face_detector.py
│   └── smartdrive_monitor.py
│
├── README.md
└── requirements.txt
```

## ⚙️ How It Works

1. The webcam captures the driver's face.
2. MediaPipe detects the face.
3. The eye region is extracted from the detected face.
4. The CNN predicts whether the eyes are open or closed.
5. Blink events and eye-closure duration are tracked.
6. If the eyes remain closed for the configured duration, SmartDrive changes the status to **DROWSY**.
7. An audio alarm is triggered.
8. Session information is stored in CSV logs for later analysis.

## ⏱️ Drowsiness Logic

SmartDrive does not classify a driver as drowsy from a single closed-eye frame.

Instead, it monitors continuous eye closure for a time threshold. This helps reduce false alarms caused by normal blinking.

## 📈 Analytics

The project records session information such as:

* Total blinks
* Maximum eye-closure duration
* Final session status
* Session timestamp

These logs can be used to visualize drowsiness-related statistics.

## 🔮 Future Improvements

* More robust eye-region detection
* Improved CNN accuracy
* Head-pose estimation
* Yawning detection
* Longer continuous webcam sessions
* Raspberry Pi integration
* Arduino-based physical buzzer
* Mobile or web dashboard
* Model optimization for edge devices

## ⚠️ Disclaimer

SmartDrive is an educational and research project. It should not be relied upon as the sole safety system for driving or other safety-critical applications.

## 👨‍💻 Project

**SmartDrive — Real-Time Driver Drowsiness Detection**

Built using computer vision and deep learning.

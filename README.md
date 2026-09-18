# SmartDrive - Real-Time Driver Drowsiness Detection

## Overview
SmartDrive is a computer vision and deep learning based driver drowsiness detection system.

## Features
- Face detection using MediaPipe
- Open and Closed eye classification
- CNN based eye prediction
- Blink counting
- Eye closure duration monitoring
- Drowsiness detection
- Audible alarm
- Session logging
- Analytics dashboard

## System Pipeline
Webcam -> Face Detection -> Eye Detection -> CNN Prediction ->
Blink Analysis -> Drowsiness Detection -> Alarm -> Session Logs

## Machine Learning
The project uses a Convolutional Neural Network (CNN).

Eye classes:
- 0 = Closed
- 1 = Open

Input image size: 64 x 64 grayscale

## Dataset
MRL Eye Dataset

Approximately 85,000 eye images were used for training and evaluation.

## Technologies
- Python
- TensorFlow
- Keras
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Google Colab

## Project Structure
SmartDrive/
├── models/
├── logs/
├── notebooks/
├── src/
├── assets/
├── README.md
└── requirements.txt

## How It Works
1. Webcam captures the driver.
2. MediaPipe detects the face.
3. The eye region is extracted.
4. CNN predicts Open or Closed.
5. The system monitors continuous eye closure.
6. If eyes remain closed for the threshold duration, drowsiness is detected.
7. An alarm is generated.
8. Session data is saved in CSV format.

## Drowsiness Threshold
Current threshold: 2 seconds of continuous eye closure.

## Disclaimer
SmartDrive is an educational computer vision prototype.
It is not a certified automotive safety system.

## Future Improvements
- Real-time continuous webcam streaming
- Yawning detection
- Head pose detection
- Raspberry Pi deployment
- Arduino buzzer integration
- Improved eye landmark detection
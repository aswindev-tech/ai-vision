# AI Vision Assistant

An intelligent, real-time computer vision assistant designed to perceive and interact with its environment. It utilizes multiple machine learning models to detect objects, recognize faces, understand hand gestures, and provide spoken feedback about the scene.

## Features

- **Real-Time Object Detection**: Uses YOLOv8 to identify and track objects, estimating their distance (e.g., "nearby", "far").
- **Voice Feedback**: Integrates `pyttsx3` for text-to-speech scene descriptions and `speech_recognition` to listen for future capabilities.
- **Hand Gesture Controls**: Uses MediaPipe to pause (`closed fist`) or resume (`open hand`) the video feed.
- **Face Recognition**: Automatically identifies known faces from the `dataset/faces/` directory using dlib and the `face_recognition` library.
- **Crowd & Environment Analysis**: Logs detections to `vision_log.csv` and uses a custom Random Forest classifier (`crowd_model.pkl`) to determine if a scene is "Crowded". Categorizes the environment as "Indoor" or "Outdoor".
- **Danger Alerts**: Automatically and verbally warns the user if hazardous items (like knives or scissors) are detected.

## Requirements

Ensure you have Python 3.9+ installed and a working webcam.

Install the required dependencies:
```bash
pip install opencv-python pyttsx3 mediapipe SpeechRecognition face-recognition ultralytics pandas scikit-learn joblib
```

*(Note: `face_recognition` requires `dlib`. Pre-built wheels for dlib can be difficult to set up on Windows without C++ build tools, but the included pip wheel makes it easier).*

## Usage

### 1. Running the Vision Assistant

To start the real-time AI assistant, simply run:
```bash
python main.py
```
- Show an **open hand** to the camera to ensure the assistant is running.
- Show a **closed fist** to pause the detection feed.
- Press **Esc** to exit.

### 2. Training the Crowd Model

As the assistant runs, it logs data into `vision_log.csv`. You can retrain the custom "Crowd Detection" model on your latest logged data by running:

```bash
python training.py
```
This script trains a Random Forest Classifier to distinguish crowded from non-crowded scenes based on the number of people and unique objects detected. It outputs `.pkl` model file which is then loaded automatically by `main.py`.

## Directory Structure

- `main.py`: The core application script.
- `training.py`: Script to train the Random Forest crowd-detection classifier.
- `dataset/faces/`: Add `.jpg` or `.png` face images here. The file name (e.g., `dulquer.jpg`) is identified as the person's name.
- `crowd_model.pkl`: The trained crowd detection model.
- `vision_log.csv`: Logged metrics of objects seen during runtime.
- `yolov8n.pt`: (Auto-downloaded) The YOLO weights for object detection.

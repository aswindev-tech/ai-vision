# AI Vision Assistant

An intelligent real-time computer vision system that integrates object detection, face recognition, gesture control, voice feedback, and environment analysis into a single application.

## Features

- **Object Detection (YOLOv8)**  
  Detects multiple objects in real time. Identifies people, furniture, vehicles, and more.
- **Face Recognition**  
  Recognizes known faces from a dataset. Announces detected individuals using voice.
- **Gesture Control**  
  Open hand: Resume system. Closed hand: Pause system.
- **Voice Feedback**  
  Provides real-time spoken updates. Describes the scene periodically.
- **Voice Input (Speech Recognition)**  
  Captures user input through microphone.
- **Danger Detection**  
  Alerts when dangerous objects such as knife, scissors, or fire are detected.
- **Environment Classification**  
  Determines whether the scene is indoor or outdoor.
- **Crowd Detection**  
  Predicts whether the environment is crowded using a trained model.
- **Logging System**  
  Stores detection data in CSV format for analysis.
- **Performance Monitoring**  
  Displays FPS and performance warnings.

## Technologies Used

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- MediaPipe
- Face Recognition
- SpeechRecognition
- pyttsx3
- Scikit-learn (joblib)

## Project Structure

```text
project/
├── main.py
├── vision_log.csv
├── crowd_model.pkl
├── yolov8n.pt
└── dataset/
    └── faces/
        ├── person1.jpg
        └── person2.jpg
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aswindev-tech/ai-vision.git
   cd ai-vision
   ```

2. **Install dependencies:**
   ```bash
   pip install opencv-python pyttsx3 mediapipe speechrecognition face-recognition ultralytics joblib pandas scikit-learn
   ```

3. **Ensure required files are available:**
   - `yolov8n.pt`
   - `crowd_model.pkl`

4. **Add face images to:**
   - `dataset/faces/`

## Usage

Run the application:
```bash
python main.py
```

## Controls

| Action | Method |
| :--- | :--- |
| **Resume System** | Open hand |
| **Pause System** | Closed hand |
| **Exit** | Press ESC |

## Output

Live video feed with:
- Object detection labels
- Distance estimation
- Environment classification
- FPS display
- Voice announcements
- Log file: `vision_log.csv`

## Notes

- Requires a working webcam and microphone.
- Performance depends on system hardware.
- Face recognition accuracy depends on dataset quality.
- Crowd model must be pre-trained.

## Future Improvements

- Mobile application support
- Improved voice command system
- Cloud-based logging
- Multi-camera support
- Enhanced gesture recognition

## License

*(Add License Here)*

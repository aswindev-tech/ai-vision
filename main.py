import cv2
import pyttsx3
import time
import os
import threading
import queue
import mediapipe as mp
import speech_recognition as sr
import face_recognition
from ultralytics import YOLO
import csv
from datetime import datetime
import joblib

speech_queue = queue.Queue()

engine = pyttsx3.init()
engine.setProperty("rate", 160)

def _speech_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        print(f"[AI] {text}")
        engine.say(text)
        engine.runAndWait()
        speech_queue.task_done()

speech_thread = threading.Thread(target=_speech_worker, daemon=True)
speech_thread.start()

def speak(text):
    speech_queue.put(text)


voice_result = {"query": "", "listening": False}

def listen():
    if voice_result["listening"]:
        return

    def _run():
        voice_result["listening"] = True
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as mic:
                print("[Mic] Listening...")
                recognizer.adjust_for_ambient_noise(mic, duration=0.5)
                audio = recognizer.listen(mic, timeout=4, phrase_time_limit=5)
            text = recognizer.recognize_google(audio)
            print(f"[You] {text}")
            voice_result["query"] = text.lower()
        except Exception as e:
            print(f"[Listen Error] {e}")
            voice_result["query"] = ""
        finally:
            voice_result["listening"] = False

    threading.Thread(target=_run, daemon=True).start()


print("[Loading] YOLO model...")
yolo = YOLO("yolov8n.pt")

try:
    model = joblib.load("crowd_model.pkl")
    model_loaded = True
except Exception as e:
    print(f"[Model Load Warning] {e}")
    model_loaded = False


LOG_FILE = "vision_log.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "time","total_objects","person_count",
            "unique_objects","furniture_count",
            "vehicle_count","hour"
        ])


known_encodings = []
known_names = []
FACE_FOLDER = "dataset/faces"

if os.path.exists(FACE_FOLDER):
    for filename in os.listdir(FACE_FOLDER):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(FACE_FOLDER, filename)
            try:
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    known_encodings.append(encs[0])
                    known_names.append(os.path.splitext(filename)[0])
            except Exception as e:
                print(f"[Face Load Error] {e}")


mp_hands = mp.solutions.hands
hand_tracker = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

last_gesture_time = 0
GESTURE_COOLDOWN = 0.3

gesture_history = []

def detect_gesture(frame):
    global last_gesture_time, gesture_history

    if time.time() - last_gesture_time < GESTURE_COOLDOWN:
        return "NONE"

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand_tracker.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
        lm = hand.landmark

        fingers = [
            lm[8].y < lm[6].y,
            lm[12].y < lm[10].y,
            lm[16].y < lm[14].y,
            lm[20].y < lm[18].y
        ]

        open_fingers = fingers.count(True)

        gesture_history.append(open_fingers)
        if len(gesture_history) > 5:
            gesture_history.pop(0)

        avg = sum(gesture_history) / len(gesture_history)
        last_gesture_time = time.time()

        cv2.putText(frame, f"Fingers:{open_fingers}", (10,200),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

        if avg >= 3:
            return "OPEN"
        elif avg <= 1:
            return "CLOSED"

    gesture_history.clear()
    return "NONE"


def get_distance_label(w):
    if w > 300:
        return "very close"
    elif w > 150:
        return "nearby"
    return "far"


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[Error] Cannot open camera.")
    exit(1)

paused = False
last_gesture_state = "NONE"

last_scene_time = 0
last_log_time = 0
last_face_time = 0

frame_count = 0
prev_time = 0

detected_objects = []

speak("AI Vision Assistant is ready.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()

    curr_time = time.time()
    fps = 1/(curr_time-prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    gesture = detect_gesture(frame)

    if gesture != "NONE" and gesture != last_gesture_state:
        if gesture == "OPEN":
            paused = False
            speak("Resuming")
        elif gesture == "CLOSED":
            paused = True
            speak("Paused")
        last_gesture_state = gesture

    cv2.putText(frame, "RUNNING" if not paused else "PAUSED",
                (10,40),0,1,(0,255,0) if not paused else (0,0,255),2)

    if paused:
        cv2.imshow("AI Vision Assistant", frame)
        if cv2.waitKey(1) == 27:
            break
        continue

    detected_objects = []

    frame_count += 1
    if frame_count % 2 == 0:
        for result in yolo(frame, verbose=False):
            for box in result.boxes:
                if box.conf[0] < 0.6:
                    continue

                label = yolo.names[int(box.cls[0])]
                detected_objects.append(label)

                x1,y1,x2,y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

                dist = get_distance_label(x2 - x1)
                cv2.putText(frame, f"{label} ({dist})", (x1,y1-8), 0, 0.6, (0,255,0), 2)

    total_objects = len(detected_objects)
    person_count = detected_objects.count("person")
    unique_objects = len(set(detected_objects))

    furniture = ["chair","sofa","tv","dining table"]
    vehicles = ["car","bus","truck","motorbike"]

    furniture_count = sum(detected_objects.count(x) for x in furniture)
    vehicle_count = sum(detected_objects.count(x) for x in vehicles)

    hour = datetime.now().hour

    # Object frequency
    object_counts = {}
    for obj in detected_objects:
        object_counts[obj] = object_counts.get(obj, 0) + 1

    if object_counts:
        top_object = max(object_counts, key=object_counts.get)
        cv2.putText(frame, f"Top: {top_object}", (10,190),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,255,200),2)

    # Danger detection
    danger_objects = ["knife","scissors","fire"]
    if any(obj in detected_objects for obj in danger_objects):
        cv2.putText(frame, "DANGER!", (300,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        speak("Warning! Dangerous object detected.")

    if model_loaded and total_objects > 0:
        pred = model.predict([[total_objects,person_count,unique_objects]])[0]
        label = "Crowded" if pred==1 else "Not Crowded"
        cv2.putText(frame,f"Scene: {label}",(10,100),0,0.7,(0,255,255),2)

    env = "Indoor" if furniture_count > vehicle_count else "Outdoor"
    cv2.putText(frame,f"Env: {env}",(10,130),0,0.7,(255,255,0),2)

    # Voice scene description
    if now - last_scene_time > 10 and total_objects > 0:
        last_scene_time = now
        speak(f"I see {person_count} persons and {unique_objects} unique objects. Environment is {env}")

    # Face recognition
    if now - last_face_time > 5 and known_encodings:
        last_face_time = now
        small = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_small)
        face_encs = face_recognition.face_encodings(rgb_small, face_locs)
        for enc in face_encs:
            matches = face_recognition.compare_faces(known_encodings, enc)
            if True in matches:
                name = known_names[matches.index(True)]
                speak(f"I recognize {name}")

    # Logging
    if total_objects > 0 and now - last_log_time > 5:
        last_log_time = now
        with open(LOG_FILE,'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%H:%M:%S"),
                total_objects,
                person_count,
                unique_objects,
                furniture_count,
                vehicle_count,
                hour
            ])

    # FPS warning
    if fps < 10:
        cv2.putText(frame, "LOW PERFORMANCE", (300,70),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    cv2.putText(frame,f"FPS:{int(fps)}",(10,160),0,0.6,(255,255,255),2)

    cv2.imshow("AI Vision Assistant", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
hand_tracker.close()

speak("Goodbye!")
speech_queue.put(None)

print("[Done]")
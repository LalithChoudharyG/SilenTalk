import base64
import cv2
import numpy as np
import mediapipe as mp
from channels.generic.websocket import AsyncWebsocketConsumer
from ultralytics import YOLO
import json
import time
import asyncio
import threading
import gc

# Add these imports
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import pyttsx3  # For text-to-speech

# Initialize pyttsx3 engine
engine = pyttsx3.init()

# Load YOLO model
model = YOLO("best.pt")  # Replace with your YOLO model path

# Class names mapping
class_names = {0: "Hello", 1: "Yes", 2: "I Love You", 3: "No", 4: "How", 5: "You",
               6: "Thank You", 7: "Eat", 8: "Sleep", 9: "Water", 10: "Beat You",
               11: "Iam", 12: "Want", 13: "Ok"}  # Corrected "I am" to "Iam"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Set target frame rate interval (for ~30 FPS)
TARGET_INTERVAL = 0.033  # in seconds (1000 ms / 30)

# Kalman filter for bounding boxes


class KalmanBoxFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 0.1, 0],
                                                [0, 0, 0, 0.1]], np.float32)
        self.initialized = False

    def update(self, x, y):
        if not self.initialized:
            self.kalman.statePre = np.array([x, y, 0, 0], np.float32)
            self.kalman.statePost = np.array([x, y, 0, 0], np.float32)
            self.initialized = True
        measurement = np.array([x, y], np.float32)
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        return int(prediction[0]), int(prediction[1])


# Initialize Kalman filters for bounding box corners
kalman_filters = [KalmanBoxFilter(), KalmanBoxFilter(),
                  KalmanBoxFilter(), KalmanBoxFilter()]

# Control garbage collection to reduce interruptions
gc.set_threshold(2000, 10, 10)

# Decode base64-encoded image


def decode_image(base64_string):
    img_data = base64.b64decode(base64_string)
    np_img = np.frombuffer(img_data, dtype=np.uint8)
    return cv2.imdecode(np_img, cv2.IMREAD_COLOR)

# Stabilize bounding box coordinates


def stabilize_bbox(bbox):
    x1, y1 = kalman_filters[0].update(bbox[0], bbox[1])
    x2, y2 = kalman_filters[1].update(bbox[2], bbox[3])
    return [x1, y1, x2, y2]

# Threaded YOLO inference function


def threaded_yolo_inference(cropped_resized_rgb):
    return model(cropped_resized_rgb, conf=0.5)

# Run hand detection and YOLO inference on the detected region


def detect_objects(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = frame.shape
    detections = []

    results = hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            x_min, y_min = min(x_list), min(y_list)
            x_max, y_max = max(x_list), max(y_list)

            x_min_pixel = int(x_min * img_width)
            y_min_pixel = int(y_min * img_height)
            x_max_pixel = int(x_max * img_width)
            y_max_pixel = int(y_max * img_height)

            margin = 20
            x_min_pixel = max(0, x_min_pixel - margin)
            y_min_pixel = max(0, y_min_pixel - margin)
            x_max_pixel = min(img_width, x_max_pixel + margin)
            y_max_pixel = min(img_height, y_max_pixel + margin)

            bbox = stabilize_bbox(
                [x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel])

            cropped_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            if cropped_img.size != 0:
                cropped_resized = cv2.resize(cropped_img, (224, 224))
                cropped_resized_rgb = cv2.cvtColor(
                    cropped_resized, cv2.COLOR_BGR2RGB)

                yolo_thread = threading.Thread(
                    target=threaded_yolo_inference, args=(cropped_resized_rgb,))
                yolo_thread.start()
                yolo_thread.join()

                yolo_results = model(cropped_resized_rgb, conf=0.5)

                if yolo_results[0].boxes:
                    detection = yolo_results[0].boxes[0]
                    class_id = int(detection.cls)
                    confidence = float(detection.conf)
                    class_name = class_names.get(class_id, "Unknown")

                    detections.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": bbox
                    })

    return detections


# Load the trained model and tokenizers
sentence_model = load_model('sentence_formation_model.h5')

with open('input_tokenizer.pkl', 'rb') as f:
    input_tokenizer = pickle.load(f)

with open('output_tokenizer.pkl', 'rb') as f:
    output_tokenizer = pickle.load(f)

with open('max_seq_lengths.pkl', 'rb') as f:
    seq_lengths = pickle.load(f)
    max_encoder_seq_length = seq_lengths['max_encoder_seq_length']
    max_decoder_seq_length = seq_lengths['max_decoder_seq_length']

# Reverse the output tokenizer for decoding
reverse_output_word_index = {
    v: k for k, v in output_tokenizer.word_index.items()}

# Function to add punctuation based on sentence structure


def add_punctuation(sentence):
    if sentence.startswith("how") or sentence.startswith("are") or sentence.startswith("do"):
        return sentence.capitalize() + "?"
    elif sentence.startswith("hello"):
        return sentence.capitalize() + ","
    elif sentence.endswith("you") or sentence.endswith("water") or sentence.endswith("eat") or sentence.endswith("sleep"):
        return sentence.capitalize() + "."
    else:
        return sentence.capitalize()

# Function to predict a sentence based on input signs


def predict_sentence(sign_sequence):
    # Convert sign sequence to integers using the input tokenizer
    input_sequence = input_tokenizer.texts_to_sequences([sign_sequence])
    input_sequence_padded = pad_sequences(
        input_sequence, maxlen=max_encoder_seq_length, padding='post')

    # Predict the output sequence
    predictions = sentence_model.predict(input_sequence_padded)
    output_sequence = np.argmax(predictions, axis=-1)

    # Decode the output sequence to words
    decoded_sentence = []
    for idx in output_sequence[0]:
        if idx == 0:
            break
        word = reverse_output_word_index.get(idx, '')
        decoded_sentence.append(word)

    # Join words and add punctuation
    sentence = ' '.join(decoded_sentence)
    sentence_with_punctuation = add_punctuation(sentence)
    return sentence_with_punctuation


class DetectionConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.camera_active = False
        self.frame_skip = 0
        self.detected_signs = []
        self.last_detection_time = time.time()
        self.current_sign = None
        self.sign_start_time = None
        self.confirmed_signs = []

    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        command = data.get("command")

        if command == "start":
            self.camera_active = True
            await self.send(text_data=json.dumps({"status": "Camera started"}))
        elif command == "stop":
            self.camera_active = False
            await self.send(text_data=json.dumps({"status": "Camera stopped"}))
            return

        if self.camera_active and data.get("frame"):
            start_time = time.time()

            if self.frame_skip % 2 == 0:
                frame_data = data.get("frame")
                frame = decode_image(frame_data)
                detections = detect_objects(frame)
                self.frame_skip = 0
            else:
                self.frame_skip += 1
                return  # Skip every other frame to reduce load

            end_time = time.time()
            processing_delay = end_time - start_time

            # Update current sign and detection time
            if detections:
                detection = detections[0]  # Assuming only one detection
                class_name = detection['class_name']
                current_time = time.time()

                if self.current_sign == class_name:
                    # Update the timer for how long the sign has been detected
                    elapsed_time = current_time - self.sign_start_time
                else:
                    # New sign detected
                    self.current_sign = class_name
                    self.sign_start_time = current_time
                    elapsed_time = 0

                # Send the timer and current sign to the frontend
                await self.send(text_data=json.dumps({
                    "current_sign": self.current_sign,
                    "elapsed_time": elapsed_time,
                    "detections": detections,
                    "processing_delay": processing_delay
                }))

                # If sign detected for at least 2 seconds, add to confirmed signs
                if elapsed_time >= 2:
                    if self.current_sign not in self.confirmed_signs:
                        self.confirmed_signs.append(self.current_sign)
                        # Reset current sign to avoid multiple additions
                        self.current_sign = None
                        self.sign_start_time = None

                    # Send confirmed signs to frontend
                    await self.send(text_data=json.dumps({
                        "confirmed_signs": self.confirmed_signs
                    }))

                self.last_detection_time = current_time
            else:
                # No detections; reset current sign and sign start time
                self.current_sign = None
                self.sign_start_time = None

            # Check if no detections for 3 seconds to form sentence
            current_time = time.time()
            if current_time - self.last_detection_time > 5 and self.confirmed_signs:
                sentence = predict_sentence(self.confirmed_signs)
                await self.send(text_data=json.dumps({
                    "sentence": sentence
                }))
                # Speak the sentence using pyttsx3
                engine.say(sentence)
                engine.runAndWait()
                self.confirmed_signs = []

            if processing_delay > 0.3:
                print(f"Spike detected: {processing_delay:.3f} seconds")

            delay = TARGET_INTERVAL - processing_delay
            if delay > 0:
                await asyncio.sleep(delay)

            if self.frame_skip % 10 == 0:
                gc.collect()

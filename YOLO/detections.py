import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

print("Starting script...")

# Initialize MediaPipe Hands
print("Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
try:
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("MediaPipe Hands initialized.")
except Exception as e:
    print(f"Error initializing MediaPipe Hands: {e}")
    exit()

# Initialize MediaPipe drawing utility
mp_draw = mp.solutions.drawing_utils

# Load your trained YOLOv8 model
print("Loading YOLOv8 model...")
try:
    model = YOLO('best.pt')  # Replace with your model's path
    print("YOLOv8 model loaded.")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    exit()

# Initialize webcam capture
print("Initializing webcam...")
cap = cv2.VideoCapture(0)  # Change the index if you have multiple webcams
if not cap.isOpened():
    print("Cannot open webcam")
    exit()
print("Webcam initialized.")

desired_size = 224  # Should match the size used during training

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read from webcam.")
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = frame.shape

    # Process frame with MediaPipe
    results = hands.process(img_rgb)

    annotated_frame = frame.copy()
    cropped_img = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the annotated frame
            mp_draw.draw_landmarks(
                annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Get bounding box coordinates
            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            x_min = min(x_list)
            y_min = min(y_list)
            x_max = max(x_list)
            y_max = max(y_list)

            # Convert normalized coordinates to pixel values
            x_min_pixel = int(x_min * img_width)
            y_min_pixel = int(y_min * img_height)
            x_max_pixel = int(x_max * img_width)
            y_max_pixel = int(y_max * img_height)

            # Expand the bounding box slightly
            box_margin = 20  # Adjust margin as needed
            x_min_pixel = max(0, x_min_pixel - box_margin)
            y_min_pixel = max(0, y_min_pixel - box_margin)
            x_max_pixel = min(img_width, x_max_pixel + box_margin)
            y_max_pixel = min(img_height, y_max_pixel + box_margin)

            # Draw bounding box on the annotated frame
            cv2.rectangle(
                annotated_frame,
                (x_min_pixel, y_min_pixel),
                (x_max_pixel, y_max_pixel),
                (0, 255, 0),
                2
            )

            # Crop the hand region
            cropped_img = frame[y_min_pixel:y_max_pixel,
                                x_min_pixel:x_max_pixel]

            # Break after the first hand (since max_num_hands=1)
            break
    else:
        # If no hand detected, skip the rest of the loop
        cv2.imshow('Annotated Frame', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Preprocess the cropped image
    if cropped_img is not None and cropped_img.size != 0:
        # Resize the cropped image to the desired size
        cropped_img_resized = cv2.resize(
            cropped_img, (desired_size, desired_size))

        # Convert image to RGB if needed (YOLOv8 expects RGB)
        cropped_img_resized_rgb = cv2.cvtColor(
            cropped_img_resized, cv2.COLOR_BGR2RGB)

        # Run YOLOv8 inference on the cropped hand image
        results = model(cropped_img_resized_rgb, conf=0.5)

        # Annotate the cropped image with detections
        annotated_cropped_img = results[0].plot()

        # Display the annotated cropped image
        cv2.imshow('Cropped Hand Detection', annotated_cropped_img)

        # Get the detection results
        detections = results[0].boxes

        # If detections are found, display the class name on the annotated frame
        if len(detections) > 0:
            # Assuming the first detection is the most confident one
            detection = detections[0]
            class_id = int(detection.cls)
            confidence = float(detection.conf)
            class_name = model.names[class_id]

            # Map True/False to 'Yes'/'No'
            if class_name == 'True':
                display_name = 'Yes'
            elif class_name == 'False':
                display_name = 'No'
            else:
                display_name = class_name

            # Display class name and confidence on the annotated frame
            cv2.putText(
                annotated_frame,
                f"{display_name} ({confidence:.2f})",
                (x_min_pixel, y_min_pixel - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )
    else:
        # If cropping failed, show a black image
        black_img = np.zeros((desired_size, desired_size, 3), dtype=np.uint8)
        cv2.imshow('Cropped Hand Detection', black_img)

    # Display the annotated frame
    cv2.imshow('Annotated Frame', annotated_frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

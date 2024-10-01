import cv2
import numpy as np
from keras.models import load_model
import mediapipe as mp

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Mediapipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

try:
    while True:
        # Grab the webcamera's image.
        ret, frame = camera.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Convert the image color to RGB (Mediapipe uses RGB)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = hands.process(image_rgb)

        # Check if hand(s) are detected
        if results.multi_hand_landmarks:
            try:
                # Optionally draw hand annotations (skip drawing landmarks to prevent errors)
                for hand_landmarks in results.multi_hand_landmarks:
                    # Instead of drawing, we just process the model prediction
                    pass

                # Resize the raw image into (224-height, 224-width) pixels for the model
                image_resized = cv2.resize(
                    frame, (224, 224), interpolation=cv2.INTER_AREA)

                # Make the image a numpy array and reshape it to the model's input shape.
                image_resized = np.asarray(
                    image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

                # Normalize the image array
                image_resized = (image_resized / 127.5) - 1

                # Model prediction
                prediction = model.predict(image_resized)
                index = np.argmax(prediction)
                class_name = class_names[index]
                confidence_score = prediction[0][index]

                # Show prediction and confidence score on the image
                label_text = f"Class: {class_name.strip()} | Confidence: {np.round(confidence_score * 100)}%"
                cv2.putText(frame, label_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            except Exception as e:
                print(f"Error during model prediction: {e}")
        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the frame in a window
        cv2.imshow("Webcam Image", frame)

        # Listen to the keyboard for presses.
        keyboard_input = cv2.waitKey(1)

        # Press 'Q' to close the window.
        if keyboard_input == ord('q') or keyboard_input == 27:  # 'Q' or ESC key
            break

except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Release the camera and close the window
    camera.release()
    cv2.destroyAllWindows()
    hands.close()
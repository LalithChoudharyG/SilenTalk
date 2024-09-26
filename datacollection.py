import cv2
import mediapipe as mp
import numpy as np
import math
import time

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Set up camera
cap = cv2.VideoCapture(0)

# Constants
offset = 20
imgSize = 300
counter = 0
folder = "Data/OK"

while True:
    success, img = cap.read()
    if not success:
        continue
    
    # Convert the image to RGB as MediaPipe requires
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the frame for hand tracking
    result = hands.process(imgRGB)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get hand bounding box coordinates
            x_min, y_min = img.shape[1], img.shape[0]
            x_max, y_max = 0, 0
            
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, cx)
                y_min = min(y_min, cy)
                x_max = max(x_max, cx)
                y_max = max(y_max, cy)

            w, h = x_max - x_min, y_max - y_min

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            try:
                imgCrop = img[y_min - offset:y_max + offset, x_min - offset:x_max + offset]
                imgCropShape = imgCrop.shape

                aspectRatio = h / w

                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                cv2.imshow('ImageCrop', imgCrop)
                cv2.imshow('ImageWhite', imgWhite)
            except Exception as e:
                print(f"Error cropping or resizing: {e}")
                continue

            # Draw the hand landmarks on the original image
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the original image with hand landmarks
    cv2.imshow('Image', img)
    
    # Handle key press events
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Image {counter} saved")

    if key == ord("q"):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()

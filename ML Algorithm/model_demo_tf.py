import cv2
import numpy as np
import mediapipe as mp
import os
from fn_model_2_1 import *


model = keras.models.load_model(r'model2_1.h5')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(model_complexity=1,
                       max_num_hands=1,
                       min_detection_confidence=0.4,
                       min_tracking_confidence=0.2)

while cap.isOpened():
    success, image = cap.read()
    n = []  # Empty list to store coordinates in and convert to tensor for evaluation
    if not success:
        print("Exiting Process. Capture was unsuccessful.")  # If no integrated camera exists
        break

    if cv2.waitKey(33) & 0xFF == ord('q'):  # Press 'q' to exit the program
        cap.release()
        cv2.destroyAllWindows()
        break

    cv2.imshow('Raw Image Window', image)
    results = hands.process(image)
    if results.multi_handedness:
        for hand in results.multi_handedness:
            if hand.classification[0].label == "Right":
                image = cv2.flip(image, 1)
                results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_pos, y_pos = [], []  # n will be used to store and transfer values out of the for loop
            h, w, c = image.shape
            for idx, lm in enumerate(hand_landmarks.landmark):
                x_pos.append(int(lm.x * w))
                y_pos.append(int(lm.y * h))

            x_max, x_min = int(max(x_pos)), int(min(x_pos))
            y_max, y_min = int(max(y_pos)), int(min(y_pos))

            crop_left = max(int(x_min - 40), 0)
            crop_right = min(int(x_max + 40), w)
            crop_top = max(int(y_min - 40), 0)
            crop_bot = min(int(y_max + 40), h)

            adjusted_x = [int(500 * ((x - crop_left) / (crop_right - crop_left))) for x in x_pos]
            adjusted_y = [int(500 * ((y - crop_top) / (crop_bot - crop_top))) for y in y_pos]
            for i in range(len(adjusted_x)):  # Stores the adjusted x and y values in coordinate pairs
                n.append(adjusted_x[i])
                n.append(adjusted_y[i])

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

            image = image[crop_top:crop_bot, crop_left:crop_right]
            image = cv2.resize(image, (500, 500))

        cv2.imshow('Annotated Image Window', image)
        predict = np.argmax(model.predict(np.array([n]), batch_size=1, verbose=0))
        print(int(predict))

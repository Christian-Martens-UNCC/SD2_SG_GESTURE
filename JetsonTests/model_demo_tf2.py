import cv2
import numpy as np
import mediapipe as mp
import os

import time

from fn_model_2_1 import *


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

model = keras.models.load_model('model2_1.h5', compile=False)

mp_drawing = mp.solutions.drawing_utils

mp_hands = mp.solutions.hands

cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=500, height=500, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink' , cv2.CAP_GSTREAMER)

hands = mp_hands.Hands(max_num_hands=1,
                       min_detection_confidence=0.4,
                       min_tracking_confidence=0.2)

timer = 0

while cap.isOpened():
    success, image = cap.read()

    h,w,c = image.shape
    n = []  # Empty list to store coordinates in and convert to tensor for evaluation
    if not success:
        print("Exiting Process. Capture was unsuccessful.")  # If no integrated camera exists
        break

    if cv2.waitKey(5) & 0xFF == ord('q'):  # Press 'q' to exit the program
        cap.release()
        cv2.destroyAllWindows()
        break


    results = hands.process(image)
    if results.multi_handedness:
        for hand in results.multi_handedness:
            if hand.classification[0].label == "Right":
#                image = cv2.flip(image, 1)
                results = hands.process(image)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:
            x_pos, y_pos = [], []  # n will be used to store and transfer values out of the for loop

            for idx, lm in enumerate(hand_landmarks.landmark):
                x_pos.append(int(lm.x * w))
                y_pos.append(int(lm.y * h))

            for i in range(len(x_pos)):
                n.append(x_pos[i])
                n.append(y_pos[i])

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)

        predict = np.argmax(model.predict(np.array([n]), batch_size=1, verbose=0))
        
        print(int(predict))

    timer1 = time.time()
    fps =1/(timer1-timer)
    timer = timer1

    cv2.putText(image,f'FPS:{int(fps)}',(50,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,0),3)
    cv2.imshow('Annotated Image Window', image)
    
    

import mediapipe as mp
import cv2
import time
import numpy as np
import os

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=600, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink' , cv2.CAP_GSTREAMER) # For jetson Nano to run camera
cap = cv2.VideoCapture(0) # To run camera on main pc

timer = 0

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            print("Exiting Process. Capture was unsuccessful.")
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR -> RGB
        
        image.flags.writeable = False # Set flag noted to improve frames
        
        results = hands.process(image) # Detection of hands in image frame
        
        image.flags.writeable = True # Set flag to true
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # RGB -> BGR
        
#        print(results) # seeing only if there is results terminal will be constanstly printing

        if results.multi_handedness:
            for hand in results.multi_handedness:
                if hand.classification[0].label == "Right":
#                    image = cv2.flip(image, 1)
                    results = hands.process(image)
                    
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand,
                                          mp_hands.HAND_CONNECTIONS,
                                          mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                          mp_drawing.DrawingSpec(color=(0, 153, 153), thickness=2, circle_radius=2))
            
        timer1 = time.time()
        fps =1/(timer1-timer)
        timer = timer1

        cv2.putText(image,f'FPS:{int(fps)}',(50,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,0),3)
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

cap.release()
cv2.destroyAllWindows()
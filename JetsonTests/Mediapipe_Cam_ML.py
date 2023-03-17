import mediapipe as mp
import cv2
import time
import numpy as np
import os
import torch
from torch import nn

# ANN() defined. Placed here for testing model | Must change values for this particular model(s) | TabNet and Classical uses different set up.
class ANN(nn.Module): 
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(42, 64)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 128)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 32)
        self.act3 = nn.ReLU()
        self.out = nn.Linear(32, 12)
  
    def forward(self,x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.out(x)
        return x

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')    # Used to clear the Running Window for cleaner demo
    
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

model = ANN()
test = torch.load('Model3_NC.params')
# Model1: Mislabeling 2,3,6,7
# Model2: Mislabeling 0,1,3,5,6,7,8,9,10 very bad
# Model3: Minor Mislabeling of 2 and sometimes 0
# Model4: Minor Mislabeling of 2,3
# Model5: Have to make work (redo save)
# Model6: Have to make work (redo save)
# Model7: Have to make work (redo save)
model.load_state_dict(test)


saved_gestures = []     # List of saved gestures for the demo
last_save = []      # List of the last saved gestures
gest_history = []   # List of previous guesses by the network. How we get more accurate results
last_gesture = -1   # The gesture that is going to be checked to either append or alter the saved gestures
n = []


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

            
#----------- Copied to see if gestures are labeled correctly ---------------------------------
# Error: mat1 and mat2 shapes cannot be multiplied (1x0 and 42x64)
        if not results.multi_hand_landmarks and last_gesture == -1: # No increase of frames.
            clear_screen()
            print("No Hand Gesture Detected")
            
        elif not results.multi_hand_landmarks and last_gesture == 10:
            clear_screen()
            print("Submitting Gesture History: ", saved_gestures)
            last_save = saved_gestures
            last_gesture = -1
            gest_history = []
            saved_gestures = []
            
        elif not results.multi_hand_landmarks and last_gesture == 11:
            clear_screen()
            print("Deleting Previous Gesture from History.")
            saved_gestures.pop()
            last_gesture = -1
            gest_history = []
            
        elif not results.multi_hand_landmarks:
            clear_screen()
            print("Saving Gesture")
            saved_gestures.append(last_gesture)
            last_gesture = -1
            gest_history = []
            
        else:
            clear_screen()
            n_tensor = torch.Tensor(n)
            n_mean = torch.mean(n_tensor, dim=0).unsqueeze(-1)
            n_var = torch.var(n_tensor, dim=0).unsqueeze(-1)
            n_normal = (n_tensor - n_mean) / torch.sqrt(n_var)
            output = torch.argmax(model(n_normal.view(1, -1))).item()
            gest_history.insert(0, output)
            print(f"Recognized Gesture: {output}")
            
            while len(gest_history) > 10:
                gest_history = gest_history[:-1]
            last_gesture = max(set(gest_history), key=gest_history.count)
            
            print(f"Gesture that's going to be saved: {last_gesture}")
            
        if saved_gestures:
            print("Saved Gestures: ", saved_gestures)
            
        if last_save:
            print("Previous Save: ", last_save)        
        

cap.release()
cv2.destroyAllWindows()
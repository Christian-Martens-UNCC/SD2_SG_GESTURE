import cv2
import mediapipe as mp
import os
import time
from fn_model_1_3 import *

class ANN(nn.Module): # ANN() not defined from fn_model_1_3. Placed here for testing model
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

#--------------- Alternative method of calling camera for Jetson Nano --------------------
# def gstreamer_pipeline(sensor_id=0,   # alternative method to setting up camera for Nano
#                        capture_width=1920,
#                        capture_height=1080,
#                        display_width=960,
#                        display_height=540,
#                        framerate=30,
#                        flip_method=0):
#     
#     return ("nvarguscamerasrc sensor-id=%d !"
#             "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
#             "nvvidconv flip-method=%d ! "
#             "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
#             "videoconvert ! "
#             "video/x-raw, format=(string)BGR ! appsink"
#             % (sensor_id, capture_width, capture_height, framerate, flip_method, display_width, display_height))
#---------- Alternative method of calling camera for Jetson Nano --------------------


# model = init_model()   # Create an identical model framework to the one that trained on the model
model = ANN()

# test = torch.load('nn_model_1-3.params')  # Load the saved model parameters from the trained model
test = torch.load('Model3_NC.params') # NC -> CPU for PC testing | Model3.params is with CUDA
# Model1: Mislabeling 2,3,6,7
# Model2: Mislabeling 0,1,3,5,6,7,8,9,10 very bad
# Model3: Minor Mislabeling of 2 and sometimes 0
# Model4: Minor Mislabeling of 2,3
# Model5: Have to make work (redo save)
# Model6: Have to make work (redo save)
# Model7: Have to make work (redo save)
model.load_state_dict(test)     # Add in the trained state to the model framework, which is now an operational model

mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles # not available in 0.8.5 of mediapipe, results in error.
mp_hands = mp.solutions.hands

#--------------- Starting up camera -------------------------------------------------
# cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=600, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink' , cv2.CAP_GSTREAMER) # For jetson Nano to run camera

# cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0))   # Capture video

cap = cv2.VideoCapture(0)   # Capture video

#--------------- Starting up camera --------------------------------------------------

saved_gestures = []     # List of saved gestures for the demo
last_save = []      # List of the last saved gestures
gest_history = []   # List of previous guesses by the network. How we get more accurate results
last_gesture = -1   # The gesture that is going to be checked to either append or alter the saved gestures

# model_complexity=1, # not available in 0.8.5 of mediapipe, results in error.

startTime = 0

hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.4, min_tracking_confidence=0.2)  # Init hand tracking model

while cap.isOpened():
    
    success, image = cap.read()
    
    n = []  # Empty list to store coordinates in and convert to tensor for evaluation
    
    if not success:
        print("Exiting Process. Capture was unsuccessful.")     # If no integrated camera exists
        break
        

#    cv2.imshow('Raw Image Window', image)   # Shows the camera POV
    
    results = hands.process(image)
    
    if results.multi_handedness:
        for hand in results.multi_handedness:
            if hand.classification[0].label == "Right":
                image = cv2.flip(image, 1)
                results = hands.process(image)
                
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_pos, y_pos = [], []    # n will be used to store and transfer values out of the for loop
            
            h, w, c = image.shape
            
            for id, lm in enumerate(hand_landmarks.landmark):
                x_pos.append(int(lm.x * w))
                y_pos.append(int(lm.y * h))
                
            x_max, x_min = int(max(x_pos)), int(min(x_pos))
            y_max, y_min = int(max(y_pos)), int(min(y_pos))
            
            crop_left = max(int(x_min - 40), 0)
            crop_right = min(int(x_max + 40), w)
            crop_top = max(int(y_min - 40), 0)
            crop_bot = min(int(y_max + 40), h)
            
            # Alters the x and y coordinates of each landmark to the cropped image's corresponding pixel
            adjusted_x = [int(500 * ((x - crop_left) / (crop_right - crop_left))) for x in x_pos]
            adjusted_y = [int(500 * ((y - crop_top) / (crop_bot - crop_top))) for y in y_pos]
            
            for i in range(len(adjusted_x)):    # Stores the adjusted x and y values in coordinate pairs
                n.append(adjusted_x[i])
                n.append(adjusted_y[i])
                
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
            
            image = image[crop_top:crop_bot, crop_left:crop_right]
            image = cv2.resize(image, (500, 500))
            
#        cv2.imshow('Annotated Image Window', image) # Not needed
        
    timer1 = time.time()
    fps =1/(timer1-startTime)
    startTime = timer1

    cv2.putText(image,f'FPS:{int(fps)}',(50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(255,0,0),3)
    cv2.imshow('Raw Image Window', image)   # Shows the camera POV | this will have the image cropped affect if
    
    if cv2.waitKey(5) & 0xFF == ord('q'):   # Press 'q' to exit the program
        cap.release()
        cv2.destroyAllWindows()
        break
    
        
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


#------------------------------------------
# command prompt to launch camera
# gst-launch-1.0 nvarguscamerasrc ! 'video/x-raw(memory:NVMM),width=600, height=480, framerate=21/1, format=NV12' ! nvvidconv flip-method=0 ! 'video/x-raw,width=960, height=616' ! nvvidconv ! nvegltransform ! nveglglessink -e

# To do:
# > check framerate setting on videocapture
# > check video/x-raw changes
# > check appsink to appsink max-buffers=1 drop=True
# > pytorch
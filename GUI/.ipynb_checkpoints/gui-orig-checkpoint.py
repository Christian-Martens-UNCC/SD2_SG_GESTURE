# https://www.pysimplegui.org/en/latest/#jump-start
import PySimpleGUI as sg
import cv2

import mediapipe as mp
from fn_model_2_1 import *
import numpy as np
import os
import time

# Define main layout for GUI
main_layout = [
    [sg.Column([
        [sg.Text('', justification='center', expand_x=True)],
        [sg.Text('', justification='center', expand_x=True)],
        [sg.Text('', justification='center', expand_x=True)],
        [sg.Image(source='', key='image', expand_y=True, size=(960, 1080))]
    ], element_justification='c', vertical_alignment='center'),
    sg.Column([
        [sg.Text('SG_GESTURE: Needle Check', font='_ 24 bold', justification='center', expand_x=True)],
        [sg.Text('Instructions', font='_ 14 underline', justification='center', expand_x=True)],
        [sg.Text('Step 1: Perform roundness measurement with 5 needles', justification='l', font='_ 16', expand_x=True)],
        [sg.Text('Step 1a: Show the adjustment gesture if an adjustment was made', justification='l', font='_ 14', expand_x=True)],
        [sg.Text('Step 2: Perform inner diameter measurement with 5 needles', justification='l', font='_ 16', expand_x=True)],
        [sg.Text('Step 2a: Show the adjustment gesture if an adjustment was made', justification='l', font='_ 14', expand_x=True)],
        [sg.Text('Step 3: Perform outer diameter measurement with 5 needles', justification='l', font='_ 16', expand_x=True)],
        [sg.Text('Step 3a: Show the adjustment gesture if an adjustment was made', justification='l', font='_ 14', expand_x=True)],
        
        [sg.Text('', justification='center', expand_x=True)],
        [sg.Text('', justification='center', expand_x=True)],
        
        [sg.Text('Recognized Gesture', justification='c', font='_ 16 underline', expand_x=True)],
        [sg.Text('', key='gesture', justification='c', font='_ 70 bold', expand_x=True)]
    ], element_justification='c', vertical_alignment='center')]
]

# Define settings layout for GUI
# Contains: Icons for each setting (like brightness, gestures, contrast)
setting_layout = [
    # NOT FINISHED
]

# Layout for GUI that controls visibilty of each sub-layout
layout = [
    [sg.Column(main_layout, key='main'), sg.Column(setting_layout, key='setting', visible=False)],
    [sg.Button('Main'), sg.Button('Setting')]
]

def mse(img1, img2):
   h, w = img1.shape
   diff = cv2.subtract(img1, img2)
   err = np.sum(diff**2)
   mse = err/(float(h*w))
   return mse

def resize_image(img, scale_percent) :
    # Calculate new size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # Resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

# Create the Window
window = sg.Window('SG_GESTURE', layout, size=(1920, 1080))
run_model = False
nano_cam = False

if nano_cam:
    cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink' , cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture(0)
    
model = keras.models.load_model('../JetsonTests/model2_1.h5', compile=False)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.2, min_tracking_confidence=0.4)

seq_frames = []

while True:
    event, values = window.read(timeout=20)

    # Read the camera frame
    ret, frame = cap.read()
    h, w, c = frame.shape
    xpos, ypos = [], []
    n = []

    # Convert the frame from BGR to RGB
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(frame)
    if event in ('Stop', sg.WIN_CLOSED, 'Close'):
        if run_model : 
            run_model = False # Stop running
            cap.release() # Release video
            if event != sg.WIN_CLOSED : window['image'].update(filename='') # Destroy picture
        # When close window or press Close
        if event in (sg.WIN_CLOSED, 'Close'): break
    # Run Model
    
    if event == 'Setting':
        window['setting'].update(visible=True)
        window['main'].update(visible=False)
    elif event == 'Main':
        window['main'].update(visible=True)
        window['setting'].update(visible=False)
    
    if run_model:
        if ret:
            if results.multi_hand_landmarks:
                for hlms in results.multi_hand_landmarks:
                    for idx, lm in enumerate(hlms.landmark):
                        xpos.append(int(lm.x*w))
                        ypos.append(int(lm.y*h))

                    for i in range(len(xpos)):
                        n.append(xpos[i])
                        n.append(ypos[i])

                    mp_drawing.draw_landmarks(frame, hlms,
                                              mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(0, 153, 153), thickness=2, circle_radius=2))

                predict = np.argmax(model.predict(np.array([n]), batch_size=1, verbose=0))
        
                window['gesture'].update(value=(str(predict)))
        # Update the PySimpleGUI Image object
        # window['image'].update(data=cv2.imencode('.png', frame)[1].tobytes())

        display_img = resize_image(frame, 170)
        # Show Image
        imgbytes = cv2.imencode('.png', display_img)[1].tobytes()
        window['image'].update(data=imgbytes)

        if event == 'Exit' or event == sg.WIN_CLOSED:
            cap.release()
            break
    else:
        # Capture sequential frames
        if len(seq_frames) < 1:
            seq_frames.append(frame)
        else:
            img1 = seq_frames[0]
            img2 = frame
            
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            error = mse(img1, img2)
            print(error)
            if error > 15:
                run_model = True
            
            seq_frames = []
        

cap.release()
window.close()

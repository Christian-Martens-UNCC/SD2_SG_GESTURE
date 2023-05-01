# https://www.pysimplegui.org/en/latest/#jump-start
import PySimpleGUI as sg
import cv2
import csv
import datetime

import mediapipe as mp
from fn_model_2_1 import *
from fn_char_storage import *
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
        [sg.Text('Schaeffler Needle Check Station', font='_ 24 bold', justification='center', expand_x=True)],
        [sg.Text('Instructions', font='_ 14 underline', justification='center', expand_x=True)],
        [sg.Text('Step 0: Obtain five needles from the output pile.', justification='l', font='_ 16', expand_x=True)],
        [sg.Text('Step 1: Perform outer diameter measurement', justification='l', font='_ 16', expand_x=True)],
        [sg.Text('Step 1a: Show the adjustment gesture if an adjustment was made', justification='l', font='_ 14', expand_x=True)],
        [sg.Text('Step 2: Perform 3-point roundness measurement', justification='l', font='_ 16', expand_x=True)],
        [sg.Text('Step 2a: Show the adjustment gesture if an adjustment was made', justification='l', font='_ 14', expand_x=True)],
        [sg.Text('Step 3: Perform 5-point roundness measurement', justification='l', font='_ 16', expand_x=True)],
        [sg.Text('Step 3a: Show the adjustment gesture if an adjustment was made', justification='l', font='_ 14', expand_x=True)],
        
        [sg.Text('', justification='center', expand_x=True)],
        [sg.Text('', justification='center', expand_x=True)],
        
        [sg.Text('Recognized Gesture', justification='c', font='_ 16 underline', expand_x=True)],
        [sg.Text('', key='gesture', justification='c', font='_ 70 bold', expand_x=True)],
        [sg.Text('Is this correct?', key='verify', justification='c', font='_ 50', expand_x=True, visible=False)],
        [sg.Text('', key='session_check', justification='c', font='_ 30', expand_x=True)]
    ], element_justification='c', vertical_alignment='center')]
]

btn_size = (60, 15)

# Define settings layout for GUI
# Contains: Icons for each setting (like brightness, gestures, contrast)
setting_layout = [
    # NOT FINISHED
    [sg.Column([
        [sg.Text('Settings', font='_ 24 bold', justification='center', expand_x=True)],
        [sg.Column([
            [sg.Button('1: Brightness', size=btn_size, button_color=('black', 'white'), pad=(10, 10)),
            sg.Button('2: Contrast', size=btn_size, button_color=('black', 'white'), pad=(10, 10)),
            sg.Button('3: Camera Flip', size=btn_size, button_color=('black', 'white'), pad=(10, 10))]
        ], justification='c')],
        [sg.Column([
            [sg.Button('4: Red Adjustment', size=btn_size, button_color=('black', 'white'), pad=(10, 10)),
            sg.Button('5: Green Adjustment', size=btn_size, button_color=('black', 'white'), pad=(10, 10)),
            sg.Button('6: Blue Adjustment', size=btn_size, button_color=('black', 'white'), pad=(10, 10))]
        ], justification='c')],
        [sg.Column([
            [sg.Button('7: Timer', size=btn_size, button_color=('black', 'white'), pad=(10, 10)),
            sg.Button('8: Detection Conf', size=btn_size, button_color=('black', 'white'), pad=(10, 10)),
            sg.Button('9: Tracking Conf', size=btn_size, button_color=('black', 'white'), pad=(10, 10))]
        ], justification='c')]
    ], element_justification='c', vertical_alignment='center', justification='c')]
]

bright_layout = [
    [sg.Column([
        [sg.Text('Brightness Setting', size=(20, 1), justification='center')],
        [sg.Slider(range=(1, 9), default_value=5, orientation='h', size=(120, 40))],
        [sg.Button('OK', size=(5, 1), pad=(10, (5, 10)), button_color=('black', 'white'), border_width=0)]
    ], element_justification='c', vertical_alignment='center', justification='c')]
]
contrast_layout = [
    [sg.Column([
        [sg.Text('Contrast Setting', size=(20, 1), justification='center')],
        [sg.Slider(range=(1, 9), default_value=5, orientation='h', size=(120, 40))],
        [sg.Button('OK', size=(5, 1), pad=(10, (5, 10)), button_color=('black', 'white'), border_width=0)]
    ], element_justification='c', vertical_alignment='center', justification='c')]
]
camera_layout = [
    [sg.Column([
        [sg.Text('Camera Orientation Setting', size=(20, 1), justification='center')],
        [sg.Slider(range=(1, 4), default_value=5, orientation='h', size=(120, 40))],
        [sg.Button('OK', size=(5, 1), pad=(10, (5, 10)), button_color=('black', 'white'), border_width=0)]
    ], element_justification='c', vertical_alignment='center', justification='c')]
]
red_layout = [
    [sg.Column([
        [sg.Text('Red Enhancement', size=(20, 1), justification='center')],
        [sg.Slider(range=(0, 10), default_value=5, orientation='h', size=(120, 40))],
        [sg.Button('OK', size=(5, 1), pad=(10, (5, 10)), button_color=('black', 'white'), border_width=0)]
    ], element_justification='c', vertical_alignment='center', justification='c')]
]
green_layout = [
    [sg.Column([
        [sg.Text('Green Enhancement', size=(20, 1), justification='center')],
        [sg.Slider(range=(0, 10), default_value=5, orientation='h', size=(120, 40))],
        [sg.Button('OK', size=(5, 1), pad=(10, (5, 10)), button_color=('black', 'white'), border_width=0)]
    ], element_justification='c', vertical_alignment='center', justification='c')]
]
blue_layout = [
    [sg.Column([
        [sg.Text('Blue Enhancement', size=(20, 1), justification='center')],
        [sg.Slider(range=(0, 10), default_value=5, orientation='h', size=(120, 40))],
        [sg.Button('OK', size=(5, 1), pad=(10, (5, 10)), button_color=('black', 'white'), border_width=0)]
    ], element_justification='c', vertical_alignment='center', justification='c')]
]
timer_layout = [
    # Change timer between checks, which can be easily implemented with gestures
    # Input string object that will continue to append characters (numbers in this case) until OK is pressed
]
detect_layout = [
    [sg.Column([
        [sg.Text('Detection Confidence', size=(20, 1), justification='center')],
        [sg.Slider(range=(1, 10), default_value=5, orientation='h', size=(120, 40))],
        [sg.Button('OK', size=(5, 1), pad=(10, (5, 10)), button_color=('black', 'white'), border_width=0)]
    ], element_justification='c', vertical_alignment='center', justification='c')]
]
tracking_layout = [
    [sg.Column([
        [sg.Text('Tracking Confidence', size=(20, 1), justification='center')],
        [sg.Slider(range=(1, 10), default_value=5, orientation='h', size=(120, 40))],
        [sg.Button('OK', size=(5, 1), pad=(10, (5, 10)), button_color=('black', 'white'), border_width=0)]
    ], element_justification='c', vertical_alignment='center', justification='c')]
]

# Layout for GUI that controls visibilty of each sub-layout
layout = [
    [
        sg.Column(main_layout, key='main'),
        sg.Column(setting_layout, key='setting', visible=False),
        sg.Column(bright_layout, key='bright', visible=False),
        sg.Column(contrast_layout, key='contrast', visible=False),
        sg.Column(camera_layout, key='camera', visible=False),
        sg.Column(red_layout, key='red', visible=False),
        sg.Column(green_layout, key='green', visible=False),
        sg.Column(blue_layout, key='blue', visible=False),
        sg.Column(timer_layout, key='timer', visible=False),
        sg.Column(detect_layout, key='detect', visible=False),
        sg.Column(tracking_layout, key='tracking', visible=False)
    ],
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

def convert_gesture(gesture, saved_gestures):
    if len(saved_gestures) == 0:
        if gesture == 0:
            return "None"
        elif gesture == 1:
            return "Up"
        elif gesture == 10:
            return "Down"
    elif len(saved_gestures) in (1, 2):
        if gesture == 0:
            return "No"
        elif gesture == 10:
            return "Yes"

# Create the Window
window = sg.Window('SG_GESTURE', layout, size=(1920, 1080))
run_model = False
nano_cam = False

if nano_cam:
    cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink' , cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture(0)
    
model = keras.models.load_model('model2_2.h5', compile=False)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.2, min_tracking_confidence=0.4)

seq_frames = []
saved_gestures = []  # List of saved gestures for the gauge checks
gest_history = []  # List of previous guesses by the network. How we get more accurate results
gest_len = 5    # Controls how long the gest_history list gets
last_gesture = -1  # The gesture that is going to be checked to either append or alter the saved gestures
setting_state = -1     # The current setting state that is being edited
curr_gesture = -1
verify_gesture = -1
need_verify = False

# Variables for current page check
current_main = True
current_settings = False

# CSV file variables
filename = "Needle_Checks.csv"

while True:
    event, values = window.read(timeout=20)

    # Read the camera frame
    ret, frame = cap.read()
    h, w, c = frame.shape
    x_pos, y_pos = [], []
    n = []

    # Convert the frame from BGR to RGB
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame)
    if event in ('Stop', sg.WIN_CLOSED, 'Close'):
        if run_model:
            run_model = False  # Stop running
            cap.release()  # Release video
            if event != sg.WIN_CLOSED: window['image'].update(filename='')  # Destroy picture
        # When close window or press Close
        if event in (sg.WIN_CLOSED, 'Close'): break
    # Run Model
    
    # # DEBUG:
    # event = '1: Brightness'
    
    if event == 'Setting':
        run_model = False
        window['setting'].update(visible=True)
        window['main'].update(visible=False)
    elif event == 'Main':
        run_model = True
        window['main'].update(visible=True)
        window['setting'].update(visible=False)
    elif event == '1: Brightness':
        window['bright'].update(visible=True)
        window['setting'].update(visible=False)
        
        window['main'].update(visible=False)
    elif event == '2: Contrast':
        window['contrast'].update(visible=True)
        window['setting'].update(visible=False)
        
        window['main'].update(visible=False)
    elif event == '3. Camera Flip':
        window['camera'].update(visible=True)
        window['setting'].update(visible=False)
        
        window['main'].update(visible=False)

    if run_model:
        if ret:
            if results.multi_hand_landmarks:
                for hlms in results.multi_hand_landmarks:
                    for idx, lm in enumerate(hlms.landmark):
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

                    mp_drawing.draw_landmarks(frame, hlms,
                                              mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(0, 153, 153), thickness=2, circle_radius=2))

                predict = np.argmax(model.predict(np.array([n]), batch_size=1, verbose=0))
                last_gesture, gest_history = char_storage(predict, gest_history, gest_len)
                window['gesture'].update(value=(str(predict)))
            elif last_gesture != -1 and current_main:
                # Check if it is the delete gesture
                if last_gesture == 11 and not need_verify:
                    if saved_gestures:
                        # Delete previous saved gesture
                        saved_gestures.pop()
                    else:
                        # No previous saved gestures
                        sg.popup_quick_message('No previous gestures recorded', 3)
                # Check if already verifying gesture
                elif not need_verify:
                    # Save the gesture to verify
                    curr_gesture = last_gesture
                    need_verify = True
                else:
                    # save verification gesture
                    verify_gesture = last_gesture
                last_gesture = -1
        
        if current_main:
            if need_verify:
                if len(saved_gestures) == 0:
                    if curr_gesture not in (1, 10, 0):
                        sg.popup_quick_message('Invalid gesture for OD Gauge Check', 3)
                        need_verify = False
                    else:
                        # Change label object to ask operator for verification
                        curr_gesture_str = convert_gesture(curr_gesture, saved_gestures)
                        window['gesture'].update(value=curr_gesture_str)
                        window['gesture'].update(text_color='red')
                        window['verify'].update(visible=True)

                        if verify_gesture == 10:
                            sg.popup_quick_message(f'Recieved YES gesture: Verified {curr_gesture_str} gesture', 2)
                            # Save curr_gesture to saved_gestures and restart cycle
                            saved_gestures.append(curr_gesture_str)

                            window['gesture'].update(value='')
                            window['gesture'].update(text_color='black')
                            window['verify'].update(visible=False)
                            curr_gesture = -1
                            verify_gesture = -1
                            need_verify = False
                        elif verify_gesture == 11:
                            sg.popup_quick_message('Recieved NO gesture: redo gesture', 2)
                            
                            window['gesture'].update(value='')
                            window['gesture'].update(text_color='black')
                            window['verify'].update(visible=False)
                            curr_gesture = -1
                            verify_gesture = -1
                            need_verify = False
                        elif verify_gesture in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
                            sg.popup_quick_message('Invalid verification gesture, try again', 2)
                            verify_gesture = -1
                elif len(saved_gestures) == 1:
                    if curr_gesture not in (10, 0):
                        sg.popup_quick_message('Invalid gesture for 3-point Gauge Check')
                        need_verify = False
                    else:
                        # Change label object to ask operator for verification
                        curr_gesture_str = convert_gesture(curr_gesture, saved_gestures)
                        window['gesture'].update(value=curr_gesture_str)
                        window['gesture'].update(text_color='red')
                        window['verify'].update(visible=True)

                        if verify_gesture == 10:
                            sg.popup_quick_message(f'Recieved YES gesture: Verified {curr_gesture_str} gesture', 2)
                            # Save curr_gesture to saved_gestures and restart cycle
                            saved_gestures.append(curr_gesture_str)

                            window['gesture'].update(value='')
                            window['gesture'].update(text_color='black')
                            window['verify'].update(visible=False)
                            curr_gesture = -1
                            verify_gesture = -1
                            need_verify = False
                        elif verify_gesture == 11:
                            sg.popup_quick_message('Recieved NO gesture: redo gesture', 2)
                            
                            window['gesture'].update(value='')
                            window['gesture'].update(text_color='black')
                            window['verify'].update(visible=False)
                            curr_gesture = -1
                            verify_gesture = -1
                            need_verify = False
                        elif verify_gesture in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
                            sg.popup_quick_message('Invalid verification gesture, try again', 2)
                            verify_gesture = -1
                elif len(saved_gestures) == 2:
                    if curr_gesture not in (10, 0):
                        sg.popup_quick_message('Invalid gesture for 3-point Gauge Check')
                        need_verify = False
                    else:
                        # Change label object to ask operator for verification
                        curr_gesture_str = convert_gesture(curr_gesture, saved_gestures)
                        window['gesture'].update(value=curr_gesture_str)
                        window['gesture'].update(text_color='red')
                        window['verify'].update(visible=True)

                        if verify_gesture == 10:
                            sg.popup_quick_message(f'Recieved YES gesture: Verified {curr_gesture_str} gesture', 2)
                            # Save curr_gesture to saved_gestures and restart cycle
                            saved_gestures.append(curr_gesture_str)

                            window['gesture'].update(value='')
                            window['gesture'].update(text_color='black')
                            window['verify'].update(visible=False)
                            curr_gesture = -1
                            verify_gesture = -1
                            
                            # Leave need_verify on for session verfication
                            need_verify = True
                        elif verify_gesture == 11:
                            sg.popup_quick_message('Recieved NO gesture: redo gesture', 2)
                            
                            window['gesture'].update(value='')
                            window['gesture'].update(text_color='black')
                            window['verify'].update(visible=False)
                            curr_gesture = -1
                            verify_gesture = -1
                            need_verify = False
                        elif verify_gesture in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
                            sg.popup_quick_message('Invalid verification gesture, try again', 2)
                            verify_gesture = -1
                elif len(saved_gestures) == 3:
                    # Review saved gestures and final verification
                    window['session_check'].update(value=f'Verify saved gestures: {saved_gestures}')
                    if verify_gesture == 10:
                        sg.popup_quick_message(f'Recieved YES gesture: Verified session', 2)
                        # Save saved_gestures list to csv file with date/time
                        # TODO: Code for saving will go here
                        now = datetime.datetime.now()
                        with open(filename, "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([now] + saved_gestures)

                        window['gesture'].update(value='')
                        window['gesture'].update(text_color='black')
                        window['verify'].update(visible=False)
                        window['session_check'].update(value='')
                        curr_gesture = -1
                        verify_gesture = -1
                        need_verify = False
                        saved_gestures = []
                    elif verify_gesture == 11:
                        sg.popup_quick_message('Recieved NO gesture: redo session', 2)

                        window['gesture'].update(value='')
                        window['gesture'].update(text_color='black')
                        window['verify'].update(visible=False)
                        window['session_check'].update(value='')
                        curr_gesture = -1
                        verify_gesture = -1
                        need_verify = False
                        saved_gestures = []
                    elif verify_gesture in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
                        sg.popup_quick_message('Invalid verification gesture, try again', 2)
                        verify_gesture = -1
                    
            # if not results.multi_hand_landmarks:
                
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
        # Settings Logic
        # Button Operations
        if event == '1: Brightness':
            # Brightness setting logic:
            # First: Set correct layout
            window['bright'].update(visible=True)
            window['setting'].update(visible=False)
            
            # Second: 
            
                
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
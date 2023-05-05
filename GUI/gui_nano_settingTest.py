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
import RPi.GPIO as GPIO

# Create Layout of the GUI
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
        
        [sg.Text('Saved Gestures', justification='center', font='_ 16 underline', expand_x=True)],
        [sg.Text('', key='saved', justification='c', font='_ 16', expand_x=True)],
        [sg.Text('', justification='center', expand_x=True)],
        [sg.Text('Current Gestures', justification='c', font='_ 16 underline', expand_x=True)],
        [sg.Text('', key='current', justification='c', font='_ 25 bold', expand_x=True)],
        
        [sg.Text('Recognized Gesture', justification='c', font='_ 16 underline', expand_x=True)],
        [sg.Text('', key='gesture', justification='c', font='_ 70 bold', expand_x=True)],
        [sg.Text('Is this correct?', key='verify', justification='c', font='_ 50', expand_x=True, visible=False)],
        [sg.Text('', key='session_check', justification='c', font='_ 20', expand_x=True)],
        [sg.Text('', key='sleep_time', justification='c', font='_ 18 bold', expand_x=True, visible=False)]
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
    [sg.Button('Main', font='_ 32'), sg.Button('Setting', font='_ 32')]
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

def convert_gesture(gesture, saved_gestures, need_verify):
    if gesture == 9:
        return "Go 49ers!"
    if need_verify:
        if gesture == 10:
            return "Yes"
        elif gesture == 11:
            return "No"
        else:
            return "Invalid"
    else:
        if len(saved_gestures) == 0:
            if gesture == 0:
                return "None"
            elif gesture == 1:
                return "Up"
            elif gesture == 10:
                return "Down"
            else:
                return "Invalid"
        elif len(saved_gestures) in (1, 2):
            if gesture == 0:
                return "No"
            elif gesture == 11:
                return "Delete previous"
            elif gesture == 10:
                return "Yes"
            else:
                return "Invalid"

# Create the PySimpleGUI Window
window = sg.Window('SG_GESTURE', layout, size=(1920, 1080))

# Initialize variables
run_model = False

# Set this variable True if using Jetson Nano with Raspberry Pi camera module, False if using personal webcam
nano_cam = True

# Settings variables
set_brightness = brightness_dic[str(brightness_dic["stored_val"])]
set_contrast = contrast_dic[str(contrast_dic["stored_val"])]
set_camflip = camera_flip_dic[str(camera_flip_dic["stored_val"])]
set_redshift = red_shift_dic[str(red_shift_dic["stored_val"])]
set_greenshift = green_shift_dic[str(green_shift_dic["stored_val"])]
set_blueshift = blue_shift_dic[str(blue_shift_dic["stored_val"])]
set_timer = check_timer_dic["stored_val"]
set_detect_conf = detect_con_dic[str(detect_con_dic["stored_val"])]
set_track_conf = tracking_con_dic[str(tracking_con_dic["stored_val"])]

if nano_cam:
    # cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! queue max-size-buffers=1 leaky=downstream ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True' , cv2.CAP_GSTREAMER)
    #  Settings tried: 1.5,0.5; 1.5,0.3; 1.2,0.3; 1.2,0.15; 1.2,0.0
    cap = cv2.VideoCapture(f'nvarguscamerasrc sensor-mode=4 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method={set_camflip} ! videobalance contrast={set_contrast} brightness={set_brightness} saturation=0.5 ! video/x-raw, format=(string)BGRx ! videoconvert ! queue max-size-buffers=1 leaky=downstream ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True' , cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture(0)

# Load Keras model
# model = keras.models.load_model('model20.h5', compile=True)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

model = "model2_2.tflite"
interpreter = tf.lite.Interpreter(model)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
# print(input_shape)

# Initialize MediaPipe Hands module with desired parameters
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=set_detect_conf, min_tracking_confidence=set_track_conf)

seq_frames = []
saved_gestures = []  # List of saved gestures for the gauge checks
gest_history = []  # List of previous guesses by the network. How we get more accurate results
gest_len = 5    # Controls how long the gest_history list gets
last_gesture = -1  # The gesture that is going to be checked to either append or alter the saved gestures
setting_state = -1     # The current setting state that is being edited
curr_gesture = -1
verify_gesture = -1
need_verify = False
gesture_event = 0
start_time = time.time()

afk_sleep_time = 10
afk_start_time = 0

# Variables for current page check
current_main = False
current_settings = False
current_sleep = True

# CSV file variables
filename = "Needle_Checks.csv"

# Andon Light stuff
RelayA = [21, 20, 26]
RelayB = [16, 19, 13]
Pin21 = 21
check_time = set_timer
andon_status = False
check_performed = False

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

GPIO.setup(RelayA, GPIO.OUT, initial=GPIO.LOW)
time.sleep(2)

def my_popup_quick_message(message, font='_ 20', auto_close_duration=2):
    return sg.popup_quick_message(message, font=font, auto_close_duration=auto_close_duration)

# Main event loop
while cap.isOpened():
    event, values = window.read(timeout=20)
    
    # Check for gesture-based events
    if gesture_event:
        event = gesture_event
        gesture_event = 0
    
    # Read the camera frame
    ret, frame = cap.read()
    if ret:
        h, w, c = frame.shape
    x_pos, y_pos = [], []
    n = []

    # Convert the frame from BGR to RGB
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if ret:    
        results = hands.process(frame)
    if event in ('Stop', sg.WIN_CLOSED, 'Close'):
        if run_model:
            run_model = False  # Stop running
            cap.release()  # Release video
            GPIO.output(Pin21, GPIO.LOW)
            GPIO.cleanup()
            if event != sg.WIN_CLOSED: window['image'].update(filename='')  # Destroy picture
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
        if afk_start_time == 0:
            afk_start_time = time.time()
        if ret:
            if time.time() - afk_start_time >= afk_sleep_time:
                # Sleep mode
                run_model = False
                window['image'].update(visible=False)
                current_main = False
                current_sleep = True
                if check_performed:
                    start_time = time.time()
            if results.multi_handedness:
                for hand in results.multi_handedness:
                    if hand.classification[0].label == "Right":
                        frame = cv2.flip(frame, 1)
                        results = hands.process(frame)
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

                    # Draw the hand landmarks on the original camera frame
                    mp_drawing.draw_landmarks(frame, hlms,
                                              mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                              mp_drawing.DrawingSpec(color=(0, 153, 153), thickness=2, circle_radius=2))

                # Make a prediction of the gesture using the trained model
                # predict = np.argmax(model.predict(np.array([n]), batch_size=1, verbose=0))
                input_data = np.array([n],dtype=np.float32)

                interpreter.set_tensor(input_details[0]['index'],input_data)

                interpreter.invoke()

                predict = np.argmax(interpreter.get_tensor(output_details[0]['index']))
                
                last_gesture, gest_history = char_storage(predict, gest_history, gest_len)

                 # Update 'current' label with digit and meaning of gesture
                output_string = str(predict) + ' - ' + convert_gesture(predict, saved_gestures, need_verify)
                window['current'].update(value=(output_string))
                print(predict)
                
                # Reset afk time
                afk_start_time = 0
            elif last_gesture != -1 and current_main:
                # Check if it is the delete gesture
                if last_gesture == 11 and not need_verify:
                    if saved_gestures:
                        # Delete previous saved gesture
                        my_popup_quick_message('Deleted previous gesture')
                        saved_gestures.pop()
                    else:
                        # No previous saved gestures
                        my_popup_quick_message('No previous gestures recorded')
                
                # COMMENTED OUT FOR EXPO:
                # elif last_gesture == 9 and not need_verify:
                #     gesture_event = 'Setting'
                #     current_main = False
                #     current_settings = True
                # Check if already verifying gesture
                elif not need_verify:
                    # Save the gesture to verify
                    curr_gesture = last_gesture
                    if current_main:
                        need_verify = True
                else:
                    # save verification gesture
                    verify_gesture = last_gesture
                last_gesture = -1
        
        if current_main:
            # Update current saved gestures label
            window['saved'].update(value=str(saved_gestures))
            if need_verify:
                if len(saved_gestures) == 0:
                    if curr_gesture not in (1, 10, 0):
                        my_popup_quick_message('Invalid gesture for OD Gauge Check')
                        need_verify = False
                    else:
                        # Change label object to ask operator for verification
                        curr_gesture_str = convert_gesture(curr_gesture, saved_gestures, need_verify=False)
                        window['gesture'].update(value=curr_gesture_str)
                        window['gesture'].update(text_color='red')
                        window['verify'].update(visible=True)

                        if verify_gesture == 10:
                            my_popup_quick_message(f'Recieved YES gesture: Verified {curr_gesture_str} gesture')
                            # Save curr_gesture to saved_gestures and restart cycle
                            saved_gestures.append(curr_gesture_str)

                            window['gesture'].update(value='')
                            window['gesture'].update(text_color='white')
                            window['verify'].update(visible=False)
                            curr_gesture = -1
                            verify_gesture = -1
                            need_verify = False
                        elif verify_gesture == 11:
                            my_popup_quick_message('Recieved NO gesture: redo gesture')
                            
                            window['gesture'].update(value='')
                            window['gesture'].update(text_color='white')
                            window['verify'].update(visible=False)
                            curr_gesture = -1
                            verify_gesture = -1
                            need_verify = False
                        elif verify_gesture in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
                            my_popup_quick_message('Invalid verification gesture, try again')
                            verify_gesture = -1
                elif len(saved_gestures) == 1:
                    if curr_gesture not in (10, 0, 11):
                        my_popup_quick_message('Invalid gesture for 3-point Gauge Check')
                        need_verify = False
                    else:
                        # Change label object to ask operator for verification
                        curr_gesture_str = convert_gesture(curr_gesture, saved_gestures, need_verify=False)
                        window['gesture'].update(value=curr_gesture_str)
                        window['gesture'].update(text_color='red')
                        window['verify'].update(visible=True)

                        if verify_gesture == 10:
                            my_popup_quick_message(f'Recieved YES gesture: Verified {curr_gesture_str} gesture')
                            # Save curr_gesture to saved_gestures and restart cycle
                            saved_gestures.append(curr_gesture_str)

                            window['gesture'].update(value='')
                            window['gesture'].update(text_color='white')
                            window['verify'].update(visible=False)
                            curr_gesture = -1
                            verify_gesture = -1
                            need_verify = False
                        elif verify_gesture == 11:
                            my_popup_quick_message('Recieved NO gesture: redo gesture')
                            
                            window['gesture'].update(value='')
                            window['gesture'].update(text_color='white')
                            window['verify'].update(visible=False)
                            curr_gesture = -1
                            verify_gesture = -1
                            need_verify = False
                        elif verify_gesture in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
                            my_popup_quick_message('Invalid verification gesture, try again')
                            verify_gesture = -1
                elif len(saved_gestures) == 2:
                    if curr_gesture not in (10, 0):
                        my_popup_quick_message('Invalid gesture for 3-point Gauge Check')
                        need_verify = False
                    else:
                        # Change label object to ask operator for verification
                        curr_gesture_str = convert_gesture(curr_gesture, saved_gestures, need_verify=False)
                        window['gesture'].update(value=curr_gesture_str)
                        window['gesture'].update(text_color='red')
                        window['verify'].update(visible=True)

                        if verify_gesture == 10:
                            my_popup_quick_message(f'Recieved YES gesture: Verified {curr_gesture_str} gesture')
                            # Save curr_gesture to saved_gestures and restart cycle
                            saved_gestures.append(curr_gesture_str)

                            window['gesture'].update(value='')
                            window['gesture'].update(text_color='white')
                            window['verify'].update(visible=False)
                            curr_gesture = -1
                            verify_gesture = -1
                            
                            # Leave need_verify on for session verfication
                            need_verify = True
                        elif verify_gesture == 11:
                            my_popup_quick_message('Recieved NO gesture: redo gesture')
                            
                            window['gesture'].update(value='')
                            window['gesture'].update(text_color='white')
                            window['verify'].update(visible=False)
                            curr_gesture = -1
                            verify_gesture = -1
                            need_verify = False
                        elif verify_gesture in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
                            my_popup_quick_message('Invalid verification gesture, try again')
                            verify_gesture = -1
                elif len(saved_gestures) == 3:
                    # Review saved gestures and final verification
                    window['session_check'].update(value=f'Verify saved gestures: {saved_gestures}')
                    if verify_gesture == 10:
                        my_popup_quick_message(f'Recieved YES gesture: Verified session, setting device to sleep mode', auto_close_duration = 5)
                        # Save saved_gestures list to csv file with date/time
                        # TODO: Code for saving will go here
                        now = datetime.datetime.now()
                        with open(filename, "a", newline="") as file:
                            writer = csv.writer(file)
                            writer.writerow([now] + saved_gestures)

                        window['gesture'].update(value='')
                        window['gesture'].update(text_color='white')
                        window['verify'].update(visible=False)
                        window['session_check'].update(value='')
                        curr_gesture = -1
                        verify_gesture = -1
                        need_verify = False
                        saved_gestures = []
                        
                        # Set device to sleep mode
                        run_model = False
                        window['image'].update(visible=False)
                        current_main = False
                        current_sleep = True
                        start_time = time.time()
                        
                        #Turn off andon light
                        GPIO.output(Pin21, GPIO.LOW)
                        andon_status = False
                        check_performed = True
                    elif verify_gesture == 11:
                        my_popup_quick_message('Recieved NO gesture: redo session')

                        window['gesture'].update(value='')
                        window['gesture'].update(text_color='white')
                        window['verify'].update(visible=False)
                        window['session_check'].update(value='')
                        curr_gesture = -1
                        verify_gesture = -1
                        need_verify = False
                        saved_gestures = []
                    elif verify_gesture in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
                        my_popup_quick_message('Invalid verification gesture, try again')
                        verify_gesture = -1

        # Resize the camera frame and update the GUI window with the image
        if ret:
            display_img = resize_image(frame, 170)
            imgbytes = cv2.imencode('.png', display_img)[1].tobytes()
            window['image'].update(data=imgbytes)

        # If the user clicked the "Exit" button or closed the window, stop running the model
        if event == 'Exit' or event == sg.WIN_CLOSED:
            run_model = False  # Stop running
            cap.release()  # Release video
            GPIO.output(Pin21, GPIO.LOW)
            GPIO.cleanup()
            if event != sg.WIN_CLOSED: window['image'].update(filename='')  # Destroy picture
            break
    else:
        if current_sleep:
            elapsed_time = time.time() - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            # print the elapsed time in the hours:minutes:seconds format
            window['sleep_time'].update(visible=True)
            window['sleep_time'].update(value="Elapsed Time since previous check: {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
            if elapsed_time >= check_time and not andon_status:
                GPIO.output(Pin21, GPIO.HIGH)
                check_performed = False
                andon_status = True
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
                    window['image'].update(visible=True)
                    current_sleep = False
                    current_main = True
                    afk_start_time = 0
                    
                    

                seq_frames = []

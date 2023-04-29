import PySimpleGUI as sg
import cv2

import mediapipe as mp
from fn_model_2_1 import *
import numpy as np
import os
import time

# Create Layout of the GUI
main_layout = [
    [sg.Column([
        [sg.Text('', justification='center', expand_x=True)],
        [sg.Text('', justification='center', expand_x=True)],
        [sg.Text('', justification='center', expand_x=True)],
        [sg.Image(source='', key='image', expand_y=True, size=(960, 1080))]
    ], element_justification='c', vertical_alignment='center'),
    sg.Column([
        [sg.Text('SG_GESTURE: Needle Check', font='_ 32 bold', justification='center', expand_x=True)],
        [sg.Text('Instructions', font='_ 24 underline', justification='center', expand_x=True)],
        [sg.Text('Step 1: Perform roundness measurement with 5 needles', justification='l', font='_ 20', expand_x=True)],
        [sg.Text('Step 1a: Show the adjustment gesture if an adjustment was made', justification='l', font='_ 18', expand_x=True)],
        [sg.Text('Step 2: Perform inner diameter measurement with 5 needles', justification='l', font='_ 20', expand_x=True)],
        [sg.Text('Step 2a: Show the adjustment gesture if an adjustment was made', justification='l', font='_ 18', expand_x=True)],
        [sg.Text('Step 3: Perform outer diameter measurement with 5 needles', justification='l', font='_ 20', expand_x=True)],
        [sg.Text('Step 3a: Show the adjustment gesture if an adjustment was made', justification='l', font='_ 18', expand_x=True)],
        [sg.Text('', justification='center', expand_x=True)],
        [sg.Text('', justification='center', expand_x=True)],
        [sg.Text('Recognized Gesture', justification='c', font='_ 20 underline', expand_x=True)],
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

# Create the PySimpleGUI Window
window = sg.Window('SG_GESTURE', layout, size=(1920, 1080))

# Initialize variables
run_model = True

# Set this variable True if using Jetson Nano with Raspberry Pi camera module, False if using personal webcam
nano_cam = True

if nano_cam:
    # cap = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! queue max-size-buffers=1 leaky=downstream ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True' , cv2.CAP_GSTREAMER)
    #  Settings tried: 1.5,0.5; 1.5,0.3; 1.2,0.3; 1.2,0.15; 1.2,0.0
    cap = cv2.VideoCapture('nvarguscamerasrc sensor-mode=4 ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv flip-method=2 ! videobalance contrast=1.0 brightness=0.15 ! video/x-raw, format=(string)BGRx ! videoconvert ! queue max-size-buffers=1 leaky=downstream ! video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=True' , cv2.CAP_GSTREAMER)
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
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.05, min_tracking_confidence=0.4)

# Main event loop
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
                

                # Update the GUI window with the recognized gesture
                window['gesture'].update(value=(str(predict)))

        # Resize the camera frame and update the GUI window with the image
        display_img = resize_image(frame, 170)
        imgbytes = cv2.imencode('.png', display_img)[1].tobytes()
        window['image'].update(data=imgbytes)

        # If the user clicked the "Exit" button or closed the window, stop running the model
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

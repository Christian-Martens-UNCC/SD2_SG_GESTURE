import cv2
import mediapipe as mp
import os
import torch


def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')    # Used to clear the Running Window for cleaner demo


nn = torch.nn

model = nn.Sequential(nn.Linear(42, 512),   # Create an identical model framework to the one that trained on the model
                      nn.Tanh(),
                      nn.Linear(512, 256),
                      nn.Tanh(),
                      nn.Linear(256, 64),
                      nn.Tanh(),
                      nn.Linear(64, 12),
                      nn.LogSoftmax(dim=1))
test = torch.load(r"C:\Users\ccm51\Documents\SD_SG_GESTURE\model\cnn_v1-0_2-2.params")  # Load the saved model
# parameters from the trained model
model.load_state_dict(test)     # Add in the trained state to the model framework, which is now an operational model

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)   # Capture video from the integrated camera
saved_gestures = []     # List of saved gestures for the demo
last_save = []      # List of the last saved gestures
gest_history = []   # List of previous guesses by the network. How we get more accurate results
last_gesture = -1   # The gesture that is going to be checked to either append or alter the saved gestures

with mp_hands.Hands(model_complexity=1,
                    max_num_hands=1,
                    min_detection_confidence=0.2,
                    min_tracking_confidence=0.2) as hands:  # Init hand tracking model

    while cap.isOpened():
        success, image = cap.read()
        n = []  # Empty list to store coordinates in and convert to tensor for evaluation
        if not success:
            print("Exiting Process. Capture was unsuccessful.")     # If no integrated camera exists
            break

        if cv2.waitKey(33) & 0xFF == ord('q'):   # Press 'q' to exit the program
            cap.release()
            cv2.destroyAllWindows()
            break

        cv2.imshow('Raw Image Window', image)   # Shows the camera POV
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

            cv2.imshow('Annotated Image Window', image)

        if not results.multi_hand_landmarks and last_gesture == -1:
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
            while len(gest_history) > 15:
                gest_history = gest_history[:-1]
            last_gesture = max(set(gest_history), key=gest_history.count)
            print(f"Gesture that's going to be saved: {last_gesture}")

        if saved_gestures:
            print("Saved Gestures: ", saved_gestures)

        if last_save:
            print("Previous Save: ", last_save)

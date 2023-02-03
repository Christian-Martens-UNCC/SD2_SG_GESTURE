import cv2
import mediapipe as mp
import os
import csv

main_path = r"C:\Users\ccm51\Documents\SD_SG_GESTURE"   # The main folder where all the files are found
video_path = main_path + r"\training_videos\C11R.MOV"    # The video with the data we want to train on
csv_path = main_path + r"\master_dataset.csv"   # The csv file which will be used to train the ML model
cropped_size = (500, 500)   # The cropped size of the cropped and annotated videos.
r_handed = True     # Bool for if the training video is of a right hand to flip the training image
gest_id = 11     # The hand position ID. Change to match video being trained
store_cropped_vid = True    # Bool if you want to save the resulting cropped video
cropped_path = main_path + r"\cropped_videos"     # Where all the cropped videos are going to be stored
cropped_name = "C11R_crop.mp4"  # Name for the outputted cropped video file

os.chdir(cropped_path)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

master_csv = open(csv_path, 'a', newline='')
writer = csv.writer(master_csv)
if store_cropped_vid:
    print('Creating... ' + cropped_name)
    cropped_vid = cv2.VideoWriter(filename=cropped_name,
                                  fourcc=cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                  fps=30,
                                  frameSize=cropped_size,
                                  isColor=True)

cap = cv2.VideoCapture(video_path)  # Sets the training video as the target capture window
vals = []   # Create an empty list to store x and y coord values in later for saving to CSV

with mp_hands.Hands(model_complexity=0,
                    max_num_hands=1,
                    min_detection_confidence=0.2,
                    min_tracking_confidence=0.2) as hands:  # Init hand tracking model

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Exiting Process")    # If loading separate unrelated images, use 'continue' instead of 'break'
            break

        image = cv2.flip(image, 0)     # Properly flips the image so that it is right-ways up
        if r_handed: image = cv2.flip(image, 1)
        results = hands.process(image)

        image.flags.writeable = True    # Draw the hand annotations on the image.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_pos, y_pos, n = [], [], []    # n will be used to store and transfer values out of the for loop
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
                n.append(gest_id)    # Stores the gesture ID along with the coordinate pairs
                vals.append(n)      # Saves values to the primary value list

                # Debugging
                # y_max_x_pos = x_pos[y_pos.index(max(y_pos))]    # The x coord that corresponds to the maximum y
                # y_min_x_pos = x_pos[y_pos.index(min(y_pos))]    # The x coord that corresponds to the minimum y
                # x_max_y_pos = y_pos[x_pos.index(max(x_pos))]    # The y coord that corresponds to the maximum x
                # x_min_y_pos = y_pos[x_pos.index(min(x_pos))]    # The y coord that corresponds to the minimum x
                #
                # cv2.circle(image, (y_max_x_pos, y_max), 10, (255, 255, 0), cv2.FILLED)
                # cv2.circle(image, (y_min_x_pos, y_min), 10, (255, 255, 0), cv2.FILLED)
                # cv2.circle(image, (x_max, x_max_y_pos), 10, (255, 255, 0), cv2.FILLED)
                # cv2.circle(image, (x_min, x_min_y_pos), 10, (255, 255, 0), cv2.FILLED)

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                image = image[crop_top:crop_bot, crop_left:crop_right]
                image = cv2.resize(image, cropped_size)

        cv2.imshow('MediaPipe Hands', image)
        if store_cropped_vid: cropped_vid.write(image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

continue_char = input("Video successfully cropped. Continue to converting to values? Type 'y' to continue.\n")

if continue_char != 'y':
    print("Exiting program without storing values.")

else:
    writer.writerows(vals)
    master_csv.close()
    print("Exiting program. Value storage is complete.")

import csv
import copy
import argparse
import itertools
from collections import deque
import cv2 as cv
import numpy as np
import mediapipe as mp

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type = int, default = 0)
    parser.add_argument("--width", help = 'cap width', type = int, default = 960)
    parser.add_argument("--height", help = 'cap height', type = int, default = 540)
    parser.add_argument('--use_static_image_mode', action = 'store_true')
    parser.add_argument("--min_detection_confidence",
                        help = 'min_detection_confidence',
                        type = float,
                        default = 0.7)
    parser.add_argument("--min_tracking_confidence",
                        help = 'min_tracking_confidence',
                        type = int,
                        default = 0.5)
    args = parser.parse_args()
    return args

def main():
    # Argument parsing 
    args = get_args()   
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    # Camera preparation 
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    while True:

        # Process Key 
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number = process_key(key)

        # Camera capture 
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                logging_csv(number, pre_processed_landmark_list)

        # Screen reflection
        cv.imshow('Hand Gesture Recognition', debug_image)
    cap.release()
    cv.destroyAllWindows()

def process_key(key):
    number = -1
    if 48 <= key <= 57:  # 0 - 9
        number = key - 48
    return number

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))
    
    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list

def logging_csv(number, landmark_list):
    if (0 <= number <= 9):
        #csv_path = "keypoints.csv"
        #csv_path = "D:\Faks\Master\DL\Keypoint generator\keypoints_train.csv"
        csv_path = "D:\Faks\Master\DL\Keypoint generator\keypoints_test.csv"
        #csv_path = "D:\Faks\Master\DL\Keypoint generator\keypoints_val.csv"
        #csv_path = "D:\Faks\Master\DL\Keypoint generator\keypoints_proba.csv"


        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

if __name__ == '__main__':

    main()

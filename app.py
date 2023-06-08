import argparse
import csv
import itertools
import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args

# model_save_path = 'model/keypoint_classifier/keypoint_classifier.hdf5'
model_save_path = r"C:\Users\Nikola\Python projects\Deel learning\keypoint_classifier.hdf5"
model = tf.keras.models.load_model(model_save_path)

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

    # Load keypoint_classifier_labels
    keypoint_classifier_labels = []
    with open(r"D:\Faks\Master\DL\Classifier\keypoint_classifier_label.csv",
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]

    while True:
        # Process Key (ESC: end)
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break

        # Camera capture
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):

                # Landmark calculation
                landmark_list = calc_landmark_list(image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                # Hand sign classification using Keras model
                hand_sign_id = np.argmax(model.predict(np.array([pre_processed_landmark_list])))

                cv.putText(image, keypoint_classifier_labels[hand_sign_id], (10, 20),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        cv.imshow('Hand Gesture Recognition', image)

    cap.release()
    cv.destroyAllWindows()

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    return [[min(int(landmark.x * image_width), image_width - 1),
             min(int(landmark.y * image_height), image_height - 1)]
            for _, landmark in enumerate(landmarks.landmark)]

def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]
    relative_landmark_list = [[x - base_x, y - base_y] for x, y in landmark_list]
    temp_landmark_list = list(itertools.chain.from_iterable(relative_landmark_list))
    max_value = max(map(abs, temp_landmark_list))
    normalized_landmark_list = [n / max_value for n in temp_landmark_list]
    return normalized_landmark_list


if __name__ == '__main__':
    main()

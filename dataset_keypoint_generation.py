import cv2
import mediapipe as mp
import csv
import itertools
import string
import os

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

def calc_landmark_list(image, landmarks):
    image_height, image_width = image.shape[:2]
    landmark_point = [[int(landmark.x * image_width), int(landmark.y * image_height)] for landmark in landmarks.landmark]
    return landmark_point


def pre_process_landmark(landmark_list):
    base_x, base_y = landmark_list[0]  # Base point for normalization
    temp_landmark_list = [[x - base_x, y - base_y] for x, y in landmark_list]
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list), default=1)
    return [n / max_value for n in temp_landmark_list]


def logging_csv(letter, landmark_list, csv_path='keypoint_asl_new.csv'):
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([letter] + landmark_list)


def process_images(image_files):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
        for file in image_files:
            image = cv2.imread(file)
            if image is None:
                print(f"Error: Cannot read {file}")
                continue
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            if not results.multi_hand_landmarks:
                print(f"Error: Yes")
                continue
            
            letter = os.path.basename(os.path.dirname(file))  # Extract letter from path
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                logging_csv(letter, pre_processed_landmark_list)


# Define dataset path and collect image files
DATASET_PATH = 'data/'
alphabet = list(string.ascii_uppercase) + [str(i) for i in range(1, 10)]
image_files = [os.path.join(DATASET_PATH, letter, f'{j}.jpg') for letter in alphabet for j in range(1, 51)]

# Process images
process_images(image_files)
print("Processing complete!")

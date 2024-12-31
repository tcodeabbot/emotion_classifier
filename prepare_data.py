import os

import cv2
import numpy as np

from utils import get_face_landmarks




# data_dir = '/Users/abbottubeine/Desktop/CodeArea/PORTFOLIO PROJECTS/Computer Vision/emotion_classifier/data'
data_dir = '/data'

# Check if the data directory exists
if not os.path.exists(data_dir):
    print(f"Data directory does not exist: {data_dir}")
else:
    print(f"Data directory found: {data_dir}")


output = []
for emotion_indx, emotion in enumerate(sorted(os.listdir(data_dir))):
    for image_path_ in os.listdir(os.path.join(data_dir, emotion)):
        image_path = os.path.join(data_dir, emotion, image_path_)

        image = cv2.imread(image_path)

        face_landmarks = get_face_landmarks(image)

        if len(face_landmarks) == 1404:
            face_landmarks.append(int(emotion_indx))
            output.append(face_landmarks)

np.savetxt('data.txt', np.asarray(output))
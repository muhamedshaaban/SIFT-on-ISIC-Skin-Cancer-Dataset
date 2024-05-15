import os
import cv2
import numpy as np
from pathlib import Path
import csv
import pandas as pd

def process_images(images_dir):
    image_count = 0  # Variable to count the number of processed images
    keypoints_list = []  # List to store keypoints
    descriptors_list = []  # List to store descriptors
    labels_list = []  # List to store labels

    for root, dirs, files in os.walk(images_dir):
        for file in files:
            if file.endswith('.jpeg') or file.endswith('.jpg') or file.endswith('.png'):
                image_count += 1  # Increment image count for each processed image
                image_path = os.path.join(root, file)
                label = os.path.basename(os.path.dirname(image_path))  # Extract label from directory name
                labels_list.append(label)

                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: Unable to read image '{file}'")
                    continue

                training_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # SIFT feature extraction
                sift = cv2.xfeatures2d.SIFT_create()
                keypoints, descriptors = sift.detectAndCompute(training_gray, None)

                keypoints_list.append(len(keypoints))  # Append number of keypoints
                descriptors_list.append(descriptors)  # Append descriptors

    # Write keypoints, descriptors, and labels to CSV file
    with open('features.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image', 'Label', 'Keypoints', 'Descriptors'])
        for i in range(len(keypoints_list)):
            writer.writerow([f'Image_{i+1}', labels_list[i], keypoints_list[i], descriptors_list[i].tolist()])

    # Print the total number of processed images
    print(f"Processed {image_count} images in '{images_dir}'")

    # Read CSV file into DataFrame
    df = pd.read_csv('features.csv')
    print("\nDataFrame:")
    print(df)

# Specify the directories containing the images using pathlib
train_images_dir = Path(r'C:/Users/Mohamed/Desktop/ds/Train')
test_images_dir = Path(r'C:/Users/Mohamed/Desktop/ds/Test')

# Process training images
print("Processing training images:")
process_images(train_images_dir)

# Process testing images
print("Processing testing images:")
process_images(test_images_dir)
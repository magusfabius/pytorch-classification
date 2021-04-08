# Author: Fabius
#use this script to check the dataset

import os

# SET THE TRAIN PATH 
DATASET_PATH = "X:/code/MyTensor_1/data/DATASETS/training_data/train"

total_classes = len(os.listdir(DATASET_PATH))
total_images = 0

for classname in os.listdir(DATASET_PATH):
    class_path = os.path.join(DATASET_PATH, classname)

    class_images = len(os.listdir(class_path))

    print(classname, " - ", class_images)
    total_images = total_images + class_images

print("TOTAL CLASSES: ", total_classes)
print("AVERAGE IMAGE PER CLASS: ", total_images/total_classes)
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from PIL import Image
import time
from datetime import date
import os
import copy
from pathlib import Path

#TODO: ARGPARSE --device --image --folder ...

# SETUP
#device torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("DEVICE in use ", device)

LABELS_TXT_PATH = "labels.txt"
MODEL_PATH = "pytorch_output/output_name.pth" #FIXME: set path name
FOLDER_DIR = "target/folder"                  #FIXME: set folder with images to test with inference

# the name of the folder must be the same of the label
test_folder = "./data/class_name"

# how to preprocess image to five to net
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

def load_classes_from_txt(txt_path=LABELS_TXT_PATH):
    with open(txt_path) as f:
        contents = f.read()
        contents = contents[:-1]
        splitted_contents = contents.split(",")
        return splitted_contents

# Visualizing the model predictions
def visualize_model(model, num_images=6):
    print("MODEL: ", model)
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


class_names = load_classes_from_txt()


def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image


def single_inference(model, image_path, data_transforms=data_transforms):
    start_time = time.time() #STATISTIC
    print(class_names[np.argmax(model(image_loader(data_transforms, image_path)).detach().numpy())])
    output = model(image_loader(data_transforms, image_path))
    end_time = time.time() #STATISTIC

    sm = torch.nn.Softmax()
    probabilities = sm(output) 
    
    # TAKE TOP 5
    top5_probs, top5_labels = probabilities.topk(5)

    # TENSOR to NUMPY
    np_probabilities = top5_probs.detach().numpy()[0]
    np_labels = top5_labels.detach().numpy()[0]  

    for i in range(0, 4):
        top_class = class_names[np_labels[i].astype(int)]
        top_prob = np_probabilities[i].astype(float)
        print(i, top_class, top_prob*100, " || ", end_time - start_time, " sec")
    

model_ft = models.resnet18().to(device)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft.load_state_dict(torch.load(MODEL_PATH))
model_ft.eval().to(device)


for filename in os.listdir(FOLDER_DIR):
    try:
        print(filename)
        image_path = os.path.join(FOLDER_DIR, filename)
        single_inference(model_ft, image_path)
    except Exception as e:
        print("Maybe it's not an image ... ", e)





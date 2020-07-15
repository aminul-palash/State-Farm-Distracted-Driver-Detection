from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import cv2
import glob
from PIL import Image
import time
import os
import copy
plt.ion()   
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

classes  = [ 
'safe driving',
'texting - right',
'talking on the phone - right',
'texting - left',
'talking on the phone - left',
'operating the radio',
'drinking',
'reaching behind',
'hair and makeup',
'talking to passenger' 
]

trans = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
def test(path,model):
    model.eval()
    img_cv = cv2.imread(path)
    image = trans(Image.fromarray(img_cv))
    image = image.view(1,3,224,224)
    
    with torch.no_grad():
        arr_predict = model(image.to(device))
        _, pred = torch.max(outputs, 1)
    ans = pred.data.cpu().numpy()[0]
    return ans

if __name__ == "__main__":
    image_path = 'data/sample.jpg'

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    model.load_state_dict(torch.load('pretrained_model/resnet.pt',map_location=torch.device('cpu'))) 

    result = test(image_path,model)

    print("Divers activity : ", classes[result])
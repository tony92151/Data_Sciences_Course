import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys
import time
import numpy as np
import math
import pandas as pd
from PIL import Image, ImageOps, ImageFilter
from datetime import datetime
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import random
import datetime
import os

from sklearn import preprocessing 




def creatData(train,lable_length,spre = 0):
    train = np.array(train)
    train_length = len(train)
    trainC_data = []
    trainT_data = []
    for t in range(train_length):
        v = np.zeros(lable_length)
        #print(train[t,1])
        for s in train[t,1].split(" "):
            #print(s)
            v[int(s)] = 1
        trainC_data.append([train[t,0],v[:spre]])
        trainT_data.append([train[t,0],v[spre:]])
    return np.array(v),np.array(trainC_data),np.array(trainT_data)


class trainDataset(Dataset):
    def __init__(self, train_lib, transform):
        self.filenames = train_lib[:,0]
        self.labels = train_lib[:,1]
        self.transform = transform
        #self.new_feature = 

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open("train/"+format(self.filenames[idx])+'.png')  # PIL image
        image2 = image.filter(ImageFilter.FIND_EDGES)
        image = self.transform(image)
        #image2 = image.filter(ImageFilter.FIND_EDGES)
        image2 = self.transform(image2)
        return image,image2, self.labels[idx]

class testDataset(Dataset):
    def __init__(self, test_lib, transform):
        test_lib = np.array(test_lib)
        self.filenames = test_lib[:,0]
        #self.labels = test_lib[:,1]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open("test/"+format(self.filenames[idx])+'.png')  # PIL image
        image2 = image.filter(ImageFilter.FIND_EDGES)
        image = self.transform(image)
        
        image2 = self.transform(image2)
        return image,image2,self.filenames[idx]
    

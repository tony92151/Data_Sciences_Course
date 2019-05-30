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

from build_data_loader import creatData,trainDataset,testDataset



os.chdir("../../../kaggleData/data")
os.getcwd()

train = pd.read_csv("train.csv")
lable = pd.read_csv("labels.csv")
test = pd.read_csv("sample_submission.csv")

lable_length = len(lable)
train_length = len(train)
test_length = len(test)
print("train length: "+format(train_length))
print("lable length: "+format(lable_length))
print("test length: "+format(test_length))


#print(np.array(lable)[397])
#print(np.array(lable)[398])
c_length = len(np.array(lable)[:398])
t_length = len(np.array(lable)[398:])
print(c_length)
print(t_length)
    
    
train_transformer = transforms.Compose([
    transforms.Resize((128,128)),              # resize the image to 
    #transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.ToTensor(),           # transform it into a PyTorch Tensor
    #transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
])   
    

trainA,train_c,train_t = creatData(train,lable_length,c_length)

trainC_dataloader = DataLoader(trainDataset(train_c, train_transformer),batch_size=32, shuffle=True)
trainT_dataloader = DataLoader(trainDataset(train_t, train_transformer),batch_size=32, shuffle=True)

test_dataloader = DataLoader(testDataset(test, train_transformer),batch_size=32,shuffle=False)

#############################################################################

class Ensemble(nn.Module):
    def __init__(self, modelA, modelB,input_length,output_length):
        super(Ensemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(input_length, output_length)
        
    def forward(self, xin,xin2):
        x1 = self.modelA(xin)
        x2 = self.modelB(xin2)
        x = torch.cat((x1, x2), dim=1)
        x = self.classifier(F.relu(x))
        return x

densenetC_model = models.densenet161(pretrained=False)
densenetC_model.classifier= nn.Linear(in_features=2208,out_features=c_length)
#resnetC_model = models.resnet50(pretrained=False)
resnetC_model = models.resnet18(pretrained=False)

resnetC_model.fc= nn.Linear(in_features=512, out_features=c_length)

model = Ensemble(densenetC_model, resnetC_model,c_length*2,c_length)
model.cuda()

#############################################################################

def train(epoch):
    for step, (x,x2,y) in enumerate(trainC_dataloader):
        data = Variable(x).cuda()   # batch x
        data2 = Variable(x2).cuda()
        target = Variable(y).cuda()   # batch y
        #print(len(np.array(target.cpu())[0]))
        #print(data.type())
        #print(target.type())
        output = model(data,data2)               # cnn output
        #loss = nn.functional.nll_loss(output, target)
        #print(len(np.array(output.cpu().detach()[0])))
        loss = loss_func(output, target.float())   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients
        #####################
        data = data.cpu()
        data2 = data2.cpu()
        target = target.cpu()
        torch.cuda.empty_cache()
        ####################
        
        if step==0:
            start = time.time()
            #break
            ti = 0
        elif step==100:
            ti = time.time()-start #total time = ti*(length/100)
            #print(ti)
            ti = ti*(len(trainC_dataloader)/100)
        if step % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.10f}\tTime Remain : {} '.
                     format(epoch, 
                            step * len(data), 
                            len(trainC_dataloader.dataset),
                            100.*step/len(trainC_dataloader), 
                            loss.data.item(),
                            datetime.timedelta(seconds=(ti*((int(len(trainC_dataloader)-step)/len(trainC_dataloader)))))))
        data.detach()   # batch x
        target.detach()   # batch y
    print("Finish")
    
#############################################################################
    
for epoch in range(5):
    if epoch==0:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00002/(2**epoch))
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.00004/(2**epoch),momentum=0.9)
    #optimizer = torch.optim.ASGD(cnn.parameters(), lr=0.001)
    #optimizer = torch.optim.Adam(cnn.parameters(), lr=0.00002/(2**epoch))
    loss_func = torch.nn.MSELoss()
    #loss_func = torch.nn.BCEWithLogitsLoss()
    #loss_func = torch.nn.MultiLabelMarginLoss()
    #loss_func = torch.nn.SmoothL1Loss()
    #loss_func = FocalLoss(class_num = lable_length)
    #optimizer = torch.optim.ASGD(cnn.parameters(), lr=0.0005/(epoch+1))
    train(epoch) #loss need <0.0045, at lest 3ep

    

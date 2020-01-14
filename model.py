"""
Created on Fri Dec 20 09:46:44 2019

@author: Sharjeel Masood
"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

training = True

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.feature_extraction = nn.Sequential(
                nn.Conv2d(1, 32, 5),
                nn.ReLU(),
                nn.Conv2d(32, 64, 5),
                nn.ReLU(),
                nn.Conv2d(64, 64, 5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 64, 5),
                nn.ReLU(),
                nn.Conv2d(64, 128, 5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                )
        self.fc1 = nn.Linear(128*18*18, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 750)
        self.fc4 = nn.Linear(750, 180)
        self.fc5 = nn.Linear(180, 5)
        '''
        self.linear_layers = nn.Sequential(
                nn.Linear(128*18*18, 1200),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(1200, 1200),
                nn.ReLU(),
                nn.Linear(1200, 120),
                nn.ReLU(),
                #nn.Dropout(p=0.5),
                nn.Linear(120, 5),
                #   nn.Softmax(dim=1)
                )
'''
    def forward(self, image):
        x = self.feature_extraction(image)
        x = x.reshape(-1, 128*18*18)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return F.softmax(x, dim=1)
   
model = CNN()    
#print(model)

    
if training is True:
    
    training_data = np.load("training_data.npy", allow_pickle=True)
    np.random.shuffle(training_data)
    np.random.shuffle(training_data)
    np.random.shuffle(training_data)
    print(len(training_data))
    
    X = torch.Tensor([i[0] for i in training_data]).view(-1,1,100,100)
    Y = torch.Tensor([i[1] for i in training_data])
    X = X/255.0

    X = X[0:500]
    Y = Y[0:500]
    #print(X.shape)
    #print(Y[0])
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_function = nn.MSELoss()
    m  = nn.LogSoftmax(dim=1)
    
    batch_size = 15
    epochs = 5
    
    #model.train()
    
    for epoch in tqdm(range(epochs)):
        running_loss = 0
        for i in tqdm(range(0, len(X), batch_size)):
            batch_x = X[i:i+batch_size]
            batch_y = Y[i:i+batch_size]
            #batch_y = batch_y.long()
            
            #print(batch_x.shape)
            #print(batch_y)
            _, pred = torch.max(batch_y, 1)
            print("\n",pred)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            _, pred01 = torch.max(outputs, 1)
            print(pred01)
            #print(outputs)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Epoch: {epoch+1}, loss: {running_loss*100}%")
    
    torch.save(model.state_dict(),"trained_model.pth")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
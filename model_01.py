"""
Created on Wed Jan  1 20:46:02 2020

@author: Sharjeel Masood
"""

import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

training = True

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.feature_extraction = nn.Sequential(
                nn.Conv2d(1, 96, 5),
                nn.ReLU(),
                nn.Conv2d(96, 256, 5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=1),
                nn.Conv2d(256, 384, 5),
                nn.ReLU(),
                nn.Conv2d(384, 384, 5),
                nn.ReLU(),
                nn.Conv2d(384, 128, 5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, 5),
                nn.ReLU(),
                nn.Conv2d(256, 256, 5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
                )
        self.linear_layers = nn.Sequential(
                nn.Linear(256*10*10, 750),
                nn.BatchNorm1d(750),
                nn.Linear(750, 180),
                nn.BatchNorm1d(180),
                nn.Linear(180, 5)
                )

    def forward(self, image):
        x = self.feature_extraction(image)
        x = x.reshape(-1, 256*10*10)
        x = self.linear_layers(x)
        return x
   
model = CNN()    
print(model)
    
if training is True:
    
    folder_path = "C:\\Users\\Dutchfoundation\\Desktop\\AI\\NFS\\self driving car-NFS\\data\\03"
    
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
    batch_size = 5
    epochs = 5
        
    for epoch in tqdm(range(epochs)):
        
        for folders in os.listdir(folder_path):
            name = os.path.join(folder_path, folders)
            
            training_data = np.load(name, allow_pickle=True)
            np.random.shuffle(training_data)
            print(len(training_data))
            
            X = torch.Tensor([i[0] for i in training_data]).view(-1,1,80,80)
            Y = torch.Tensor([i[1] for i in training_data])
            
            print(X.shape)
            running_loss = 0
            for i in tqdm(range(0, len(X), batch_size)):
                batch_x = X[i:i+batch_size]
                batch_y = Y[i:i+batch_size]
                
                _, pred = torch.max(batch_y, 1)
                pred = pred.long()
                print("\n",pred)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                
                _, pred01 = torch.max(outputs, 1)
                print(pred01)
                loss = F.cross_entropy(outputs, pred)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
        print(f"Epoch: {epoch+1}, loss: {running_loss}%")
    
    torch.save(model.state_dict(),"trained_model_01.pth")
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
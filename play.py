"""
Created on Thu Dec 26 20:36:26 2019

@author: Sharjeel Masood
"""

import torch
import cv2
import numpy as np
from PIL import ImageGrab
import time
from model_01 import CNN
from getkeys import key_check
from directkeys import PressKey, ReleaseKey, W, A, S, D 

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

def left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(S)
    ReleaseKey(D)

def right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)

def breaks():
    PressKey(S)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    
model = CNN()
model.load_state_dict(torch.load("trained_model.pth"))

print(model)


def main():
    
    last_time = time.time() 
    
    
    print("Taking Control in.. \n")
    for i in range(5):
        print(i+1)
        time.sleep(1)
        
    paused = False
    
    while(True):
        
        if not paused:
            model.eval()
            screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
            #print("loop took {} seconds".format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (80,80))
            screen_tensor = torch.Tensor(screen).view(-1,80,80)
            final_screen = screen_tensor.view(-1,1,80,80)
            
            for param in model.parameters():
                param.requires_grad = False
            predictions = model(final_screen)
            print(predictions)
            _, pred = torch.max(predictions, 1)
            print(pred)
            pred = np.int(pred)
            #print(pred)
            
            if pred == 0:
                left()
                
            if pred == 1:
                straight()
            
            if pred == 2:
                breaks()
            
            if pred == 3:
                right()
            if pred == 4:
                ReleaseKey(A)
                ReleaseKey(S)
                ReleaseKey(D)
                ReleaseKey(W)
                
        
        keys = key_check()
        
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(S)
                ReleaseKey(D)
                ReleaseKey(W)
                time.sleep(1)



main()




























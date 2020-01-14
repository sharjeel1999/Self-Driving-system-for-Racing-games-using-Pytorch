"""
Created on Fri Aug 20 09:46:44 2019

@author: Sharjeel Masood
"""

import numpy as np
from PIL import ImageGrab
import cv2
import time
from directkeys import ReleaseKey, PressKey, W, A, S, D
from getkeys import key_check
import os


def keys_to_output(keys):
    
    output = [0,0,0,0,0]
    
    if 'A' in keys:
        output[0] = 1
    elif 'D'in keys:
        output[3] = 1
    elif 'W' in keys:
        output[1] = 1
    elif 'S' in keys:
        output[2] = 1
    else:
        output[4] = 1
    

    return output
  

def screen_record(file_name):
    
    for i in list(range(5))[::-1]:
        print(i+1)
        time.sleep(1)   
    
    last_time = time.time()
    print(last_time)
    training_data = []
    i = 500
    while(i > 0):
        screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen, (80,80))
        keys = key_check()
        output = keys_to_output(keys)
        print(output)
        training_data.append([screen, output])
        
        #print("loop took {} seconds".format(time.time()-last_time))
        last_time = time.time()
        
        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.random.shuffle(training_data)
            np.save(file_name, training_data)
            i = i-500
        

paths = ['training_data.npy','training_data_01.npy','training_data_02.npy','training_data_03.npy','training_data_04.npy','training_data_05.npy','training_data_06.npy','training_data_07.npy','training_data_08.npy','training_data_09.npy','training_data_10.npy']

for i in range(len(paths)):
    file_name = paths[i]
    #training_data = []
    
    screen_record(file_name)



























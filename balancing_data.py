"""
Created on Mon Jan  6 16:36:54 2020

@author: Sharjeel Masood
"""

import numpy as np
import os

total = 0

folder_path = "C:\\Users\\Dutchfoundation\\Desktop\\AI\\NFS\\self driving car-NFS\\data\\03"

for folders in os.listdir(folder_path):
    name = os.path.join(folder_path, folders)
    
    lefts = []
    rights = []
    forwards = []
    breaks = []
    none = []

    train_data = np.load(name, allow_pickle=True)

    total  = total + len(train_data)
    '''
    np.random.shuffle(train_data)
    
    for data in train_data:
        img = data[0]
        choice = data[1]
    
        if choice == [1,0,0,0,0]:
            lefts.append([img,choice])
        elif choice == [0,1,0,0,0]:
            forwards.append([img,choice])
        elif choice == [0,0,0,1,0]:
            rights.append([img,choice])
        elif choice == [0,0,1,0,0]:
            breaks.append([img,choice])
        elif choice == [0,0,0,0,1]:
            none.append([img,choice])
        else:
            print('no matches')
        
    a = len(lefts)
    b = len(rights)
    c = len(forwards)
    d = [a,b,c]
    e = min(d)
    forwards = forwards[:e]
    rights = rights[:e]
    lefts = lefts[:e]
    
    final_data = forwards + lefts + rights + breaks + none
    print(e)
    #print(len(final_data))
    np.save(name, final_data)
    '''
        
print(total)        
        
        
        
        
        
        
        
        
        
        
        



































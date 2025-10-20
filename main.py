import os 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from FaceDataset import FaceDataset, Collate_fn

import numpy as np
import cv2


import code


batchsize= 32
learning_rate= .001

def getData():
    train_data_path = './images/train'
    train_labels_path = './labels/train'
    val_data_path = './images/val'
    val_labels_path = './labels/val'

    train_data = []
    val_data = []
    train_labels = []
    val_labels = []

    for idx, file in enumerate(os.listdir(train_data_path)):
        if idx > 1000: break
        img = cv2.imread(os.path.join(train_data_path,file))
        img = cv2.cvtColor(cv2.resize(img,(224,224)),cv2.COLOR_BGR2RGB)
        train_data.append(img)
    
    for idx, file in enumerate(os.listdir(val_data_path)):
        if idx > 1000: break
        img = cv2.imread(os.path.join(val_data_path,file))
        img = cv2.cvtColor(cv2.resize(img,(224,224)),cv2.COLOR_BGR2RGB)
        val_data.append(img)
    
    for idx, file in enumerate(os.listdir(train_labels_path)):
        if idx > 1000: break
        label= np.loadtxt(open(os.path.join(train_labels_path,file),'rb'),dtype=np.float32)
        train_labels.append(label)
    
    for idx, file in enumerate(os.listdir(val_labels_path)):
        if idx > 1000: break
        label = np.loadtxt(open(os.path.join(val_labels_path,file),'rb'),dtype=np.float32)
        val_labels.append(label)

    return np.stack(train_data), np.stack(val_data), train_labels, val_labels


def getTorchLoaders(train_data, val_data, train_labels, val_labels):
    
    train_dataset = FaceDataset(
        torch.from_numpy(train_data).permute(0,3,1,2).float() / 255.0,
        [torch.tensor(lbl,dtype=torch.float32) for lbl in train_labels ]
        )
    val_dataset = FaceDataset(
        torch.from_numpy(val_data).permute(0,3,1,2).float() / 255.0,
        [torch.tensor(lbl, dtype=torch.float32) for lbl in val_labels]
    )
    train_loader = DataLoader(train_dataset,batch_size=batchsize,shuffle=True, drop_last=True, collate_fn=Collate_fn)
    val_loader = DataLoader(val_dataset,batch_size=batchsize, shuffle=False, collate_fn=Collate_fn)

    return train_loader, val_loader










    


if __name__ == '__main__':
    train_data, val_data, train_labels, val_labels = getData()
    
    train_loader , val_loader = getTorchLoaders(train_data, val_data, train_labels, val_labels)
    



    code.interact(local=locals())



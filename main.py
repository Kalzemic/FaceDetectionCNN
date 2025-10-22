import os 
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from DetectorNet import DetectorNet
from FaceDataset import FaceDataset, Collate_fn
from DetectorLoss import DetectorLoss
import matplotlib.pyplot as plt
import numpy as np
import cv2


import code


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
epochs = 85
batchsize= 8
learning_rate= 5e-4
#l2lambda = 1e-4
#adam_betas= (.9,.99)

def getData():
    train_data_path = './images/train'
    train_labels_path = './labels/train'
    val_data_path = './images/val'
    val_labels_path = './labels/val'

    train_data = []
    val_data = []
    train_labels = []
    val_labels = []

    label_files = sorted(os.listdir(train_labels_path))

    for idx, label_file in enumerate(label_files):
        if idx > 1000: break 

        base_name= os.path.splitext(label_file)[0]
        image_file = base_name + '.jpg'

        # Load Label
        label_path = os.path.join(train_labels_path, label_file)
        label = np.loadtxt(open(label_path, 'rb'), dtype=np.float32)
        if label.ndim == 1:
            label = np.expand_dims(label, axis=0)
        
        # Load Image
        img_path = os.path.join(train_data_path, image_file)
        if not os.path.exists(img_path):
            print(f"Warning: Image {image_file} not found for label {label_file}. Skipping.")
            continue # Skip this label if the image is missing

        img = cv2.imread(img_path)
        img = cv2.cvtColor(cv2.resize(img,(224,224)),cv2.COLOR_BGR2RGB)
        
        train_data.append(img)
        train_labels.append(label)

    label_files = sorted(os.listdir(val_labels_path))

    for idx, label_file in enumerate(label_files):
        if idx > 1000: break 

        base_name= os.path.splitext(label_file)[0]
        image_file = base_name + '.jpg'

        # Load Label
        label_path = os.path.join(val_labels_path, label_file)
        label = np.loadtxt(open(label_path, 'rb'), dtype=np.float32)
        if label.ndim == 1:
            label = np.expand_dims(label, axis=0)
        
        # Load Image
        img_path = os.path.join(val_data_path, image_file)
        if not os.path.exists(img_path):
            print(f"Warning: Image {image_file} not found for label {label_file}. Skipping.")
            continue # Skip this label if the image is missing

        img = cv2.imread(img_path)
        img = cv2.cvtColor(cv2.resize(img,(224,224)),cv2.COLOR_BGR2RGB)
        
        val_data.append(img)
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





def buildTrainingTarget(labels, S=7, B=2):

    c_params = 6
    batch_size = len(labels)
    target  = torch.zeros(batch_size,B*6,S,S)

    for i,boxlist in enumerate(labels):
        for box in boxlist:
            cls, x, y, w, h = box.tolist()

            cell_x =int(x * S)
            cell_y = int(y * S)

            x_cell = x*S - cell_x #subtract floor
            y_cell = y*S - cell_y 
            # for b in range(B):
            #     idx = b*c_params
            #     target[i,idx: idx + c_params ,cell_y,cell_x] = torch.tensor([
            #         x_cell, y_cell,w ,h, 1.0, 1.0
            #     ])
            target[i,0 : c_params , cell_y, cell_x] = torch.tensor ([
                    x_cell, y_cell,w ,h, 1.0, 1.0
            ])
    return target


def trainDetectorNet(net, criterion, optimizer):
    
    trainLosses = np.zeros(epochs)
    valLosses = np.zeros(epochs)

    for epoch in range(epochs):
        
        net.train()
        batchLosses = []
        for x,y in train_loader:
            
            #preprocess data and GPU integration
            x= x.to(device)
            target = buildTrainingTarget(y)
            target = target.to(device)

            #forward
            output = net(x)

            #calc loss
            loss = criterion(output, target)

            #backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batchLosses.append(loss.item())
        
        trainLosses[epoch] = np.mean(batchLosses)

        net.eval()
        batchLosses = []
        for x,y in val_loader:
            
            x= x.to(device)
            target = buildTrainingTarget(y)
            target = target.to(device)

            output = net(x)
            loss = criterion(output,target)
            batchLosses.append(loss.item())
        
        valLosses[epoch] = np.mean(batchLosses)


       
    return net, trainLosses, valLosses



    



    


if __name__ == '__main__':
    train_data, val_data, train_labels, val_labels = getData()
    
    train_loader , val_loader = getTorchLoaders(train_data, val_data, train_labels, val_labels)



    net = DetectorNet()
    criterion = DetectorLoss(lambda_coord=3)
    optimizer = torch.optim.Adam(net.parameters(),lr=learning_rate)


    #run loss example 

    # x,y = next(iter(val_loader))
    # output = net(x)
    
    # target = buildTrainingTarget(y)

    # loss = criterion(output,target)

    net = net.to(device)

    net, trainLoss, valLoss = trainDetectorNet(net, criterion, optimizer)

    fig,axs = plt.subplots(1,2)
    axs[0].plot(range(epochs),trainLoss)
    axs[0].set_title('Train Losses by Epochs')
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[1].plot(range(epochs),valLoss)
    axs[1].set_title('Validation Losses by Epochs')
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    plt.show()

    code.interact(local=locals())



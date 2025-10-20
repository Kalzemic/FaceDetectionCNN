import torch 
import torch.nn as nn

class DetectorNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3,8,kernel_size=3,padding=1), # 224- 3 +((1 * 2) + 1) = 224
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2),strid=(2,2)), # 224 / 2 = 112
            nn.Conv2d(8,16,kernel_size=3,padding=1), # 112 - 3 + (1 * 2) + 1 = 112
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)), #112 / 2 = 56 
            nn.Conv2d(16,32,kernel_size=3,padding=1), # 56 - 3 + (1*2) + 1 = 56
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)), #56 / 2 = 28
            nn.Conv2d(32,64,kernel_size=3,padding=1), # 28
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)), # 28 / 2 = 14
            nn.Conv2d(64,128,kernel_size=3,padding=1), #14
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)) #7 
        )
    
    def forward(self,x):
        
        return x


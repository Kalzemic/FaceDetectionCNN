import torch 
import torch.nn as nn

class DetectorNet(nn.Module):
    def __init__(self, B=2, S=7):
        super().__init__()

        self.B = B
        self.S = S

        self.num_classes = 1
        self.features = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,padding=1), # 224- 3 +((1 * 2) + 1) = 224
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)), # 224 / 2 = 112
            nn.Conv2d(64,128,kernel_size=3,padding=1), # 112 - 3 + (1 * 2) + 1 = 112
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)), #112 / 2 = 56 
            nn.Conv2d(128,256,kernel_size=3,padding=1), # 56 - 3 + (1*2) + 1 = 56
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)), #56 / 2 = 28
            nn.Conv2d(256,512,kernel_size=3,padding=1), # 28
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)), # 28 / 2 = 14
            # nn.Dropout(p=.1),
            nn.Conv2d(512,1024,kernel_size=3,padding=1), #14
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)), #7
            nn.Dropout(p=.2) 

        )
    
        self.pred = nn.Conv2d(1024, 6 * self.B, kernel_size=1) # 7




    def forward(self,x):
        
        x = self.features(x)
        x = self.pred(x)
        return x


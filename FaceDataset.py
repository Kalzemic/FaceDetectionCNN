import torch
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    def __init__(self,images,labels):
        self.images= images
        self.labels= labels
    
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    
    def __len__(self):
        return len(self.images)

def Collate_fn(batch):
    imgs, targets = zip(*batch)   
    imgs = torch.stack(imgs)      
    return imgs, list(targets)


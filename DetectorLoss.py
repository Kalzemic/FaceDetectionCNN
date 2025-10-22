import torch
import torch.nn as nn 


class DetectorLoss(nn.Module):
    def __init__(self, B=2, S=7, lambda_coord=5,lambda_noobj=.5):
        super().__init__()
        self.B = B 
        self.S = S 
        self.lambda_coord= lambda_coord 
        self.lambda_noobj = lambda_noobj
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):

        batchsize = pred.shape[0]

        pred = pred.view(batchsize, self.B, 6, self.S, self.S)
        target = target.view(batchsize, self.B, 6, self.S , self.S)
        

        #get obj mask 
        obj_mask_4d = (target[..., 4, :, :] > 0)           # (8, 2, 7, 7)
        obj_mask_5d = obj_mask_4d.unsqueeze(2).expand_as(pred[..., 0:2, :, :])  # (8, 2, 2, 7, 7)

        no_obj_mask_4d = ~obj_mask_4d


        xy_loss = self.loss(pred[...,0:2,:,:][obj_mask_5d], target[...,0:2,:,:][obj_mask_5d])

        wh_loss = self.loss(
            torch.sqrt(torch.clamp(pred[...,2:4,:,:][obj_mask_5d], min=1e-6)),
            torch.sqrt(torch.clamp(target[...,2:4,:,:][obj_mask_5d], min=1e-6))
        )

        coord_loss = self.lambda_coord * (xy_loss + wh_loss)


        
        
        conf_loss_obj = self.loss( pred[...,4,:,:][obj_mask_4d], target[...,4,:,:][obj_mask_4d])
        conf_loss_noobj = self.lambda_noobj * (self.loss(pred[...,4,:,:][no_obj_mask_4d], target[... ,4,:,:][no_obj_mask_4d]))
    
        total_loss = coord_loss + conf_loss_obj + conf_loss_noobj
        return total_loss / (batchsize * self.S * self.S * self.B)

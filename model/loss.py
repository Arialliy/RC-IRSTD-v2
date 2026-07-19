import torch.nn as nn
import numpy as np
import  torch
import torch.nn.functional as F
from skimage import measure


def SoftIoULoss( pred, target):
        pred = torch.sigmoid(pred)
  
        smooth = 1

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        
        loss = (intersection_sum + smooth) / \
                    (pred_sum + target_sum - intersection_sum + smooth)
    
        loss = 1 - loss.mean()

        return loss

def Dice( pred, target,warm_epoch=1, epoch=1, layer=0):
        pred = torch.sigmoid(pred)
  
        smooth = 1

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))

        loss = (2*intersection_sum + smooth) / \
            (pred_sum + target_sum + intersection_sum + smooth)

        loss = 1 - loss.mean()

        return loss

class SLSIoULoss(nn.Module):
    def __init__(self):
        super(SLSIoULoss, self).__init__()


    def forward(self, pred_log, target,warm_epoch, epoch, with_shape=True):
        pred = torch.sigmoid(pred_log)
        if pred.dtype in (torch.float16, torch.bfloat16):
            # Pixel-count reductions can overflow at ordinary 256x256 inputs
            # in float16, while LLoss's small stabilizer can underflow.  Keep
            # the graph connected but evaluate the loss itself in float32.
            pred = pred.float()
            target = target.float()

        intersection = pred * target

        intersection_sum = torch.sum(intersection, dim=(1,2,3))
        pred_sum = torch.sum(pred, dim=(1,2,3))
        target_sum = torch.sum(target, dim=(1,2,3))
        has_target = target_sum > 0
        
        dis = torch.pow((pred_sum-target_sum)/2, 2)

        # Non-empty targets retain the original SLS ratios.  Empty target
        # crops need an explicit background objective: both ratios otherwise
        # become 0/0 once sigmoid underflows for a confidently empty output.
        # Mean foreground probability is a bounded, differentiable false-
        # positive penalty and tends to zero with an empty prediction.
        denominator_floor = torch.finfo(pred_sum.dtype).tiny
        alpha_denominator = torch.max(pred_sum, target_sum) + dis
        foreground_alpha = (
            torch.min(pred_sum, target_sum) + dis
        ) / alpha_denominator.clamp_min(denominator_floor)
        alpha = torch.where(
            has_target,
            foreground_alpha,
            torch.ones_like(foreground_alpha),
        )

        union = pred_sum + target_sum - intersection_sum
        foreground_overlap = intersection_sum / union.clamp_min(denominator_floor)
        background_overlap = 1.0 - pred.mean(dim=(1, 2, 3))
        overlap = torch.where(has_target, foreground_overlap, background_overlap)
        lloss = LLoss(pred, target)

        if epoch>warm_epoch:       
            siou_loss = alpha * overlap
            if with_shape:
                loss = 1 - siou_loss.mean() + lloss
            else:
                loss = 1 -siou_loss.mean()
        else:
            loss = 1 - overlap.mean()
        return loss
    
    

def LLoss(pred, target):
        loss = pred.sum() * 0.0

        patch_size = pred.shape[0]
        h = pred.shape[2]
        w = pred.shape[3]        
        x_index = torch.arange(0,w,1).view(1, 1, w).repeat((1,h,1)).to(pred) / w
        y_index = torch.arange(0,h,1).view(1, h, 1).repeat((1,1,w)).to(pred) / h
        smooth = 1e-8
        for i in range(patch_size):  

            # Location/shape is undefined for a target-free crop.  Such a
            # sample is handled by the background term in SLSIoULoss instead.
            sample_has_target = (target[i].sum() > 0).to(pred.dtype)

            pred_centerx = (x_index*pred[i]).mean()
            pred_centery = (y_index*pred[i]).mean()

            target_centerx = (x_index*target[i]).mean()
            target_centery = (y_index*target[i]).mean()
           
            angle_loss = (4 / (torch.pi**2) ) * (torch.square(torch.arctan((pred_centery) / (pred_centerx + smooth)) 
                                                            - torch.arctan((target_centery) / (target_centerx + smooth))))

            pred_length = torch.sqrt(pred_centerx*pred_centerx + pred_centery*pred_centery + smooth)
            target_length = torch.sqrt(target_centerx*target_centerx + target_centery*target_centery + smooth)
            
            length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)
        
            loss = loss + sample_has_target * (1 - length_loss + angle_loss) / patch_size
        
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

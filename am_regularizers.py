import torch
import numpy as np


def LPLoss(img, p=1):
    if np.isinf(p):
        lp_loss = torch.max(img)
    else:
        lp_loss = torch.sum(torch.pow(img, p))

    n_pixels_per_img = img.shape[0] * img.shape[1]
    norm_lp_loss = lp_loss / n_pixels_per_img
    
    return norm_lp_loss


def TVLoss(img):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)
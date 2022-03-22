"""
This file contains some loss functions that may be used as 
regularizers during AM visualization.

Loss functions include:
    - TV loss*
    - LP loss*
    - BCE loss on pixel valeus
    - Log loss with target value as 255
    - Sigmoid loss

*losses are standard and should be used.
"""

import torch
import torch.nn as nn
import numpy as np


def TVLoss(img):
    ori_img_h = img[:-1, :]
    ori_img_v = img[:, :-1]
    shift_img_h = img[1:, :]
    shift_img_v = img[:, 1:]

    h_loss = torch.sum(torch.square(ori_img_h - shift_img_h))
    v_loss = torch.sum(torch.square(ori_img_v - shift_img_v))

    tv_loss = h_loss + v_loss
    n_pixels = img.shape[0] * img.shape[1]
    norm_tv_loss = tv_loss / n_pixels / (14 ** 2)
    
    return norm_tv_loss


def LPLoss(img, p=1):
    if np.isinf(p):
        lp_loss = torch.max(img)
    else:
        lp_loss = torch.pow(torch.sum(torch.pow(img, p)), 6 / p)

    n_pixels_per_img = img.shape[0] * img.shape[1]
    norm_lp_loss = lp_loss / n_pixels_per_img / (80 ** 6)
    
    return norm_lp_loss


def InputBCELoss(img):
    n_pixel = img.shape[0] * img.shape[1]
    img_sum = torch.sum(img)
    target_sum = 255
    loss = torch.square(target_sum - img_sum)

    return loss / n_pixel


def InputClipLoss(img, device):
    max_pixel_value = 255
    target_img = torch.full(img.shape, max_pixel_value).to(device)
    loss_img = - torch.log(torch.abs(target_img - img))
    loss = torch.sum(loss_img)

    return loss 


def SigmoidLoss(img, target=0.9):
    n_pixels = img.shape[0] * img.shape[1]
    sigmoid_output = torch.sigmoid(img)
    target = torch.full_like(sigmoid_output, target)
    loss_func = nn.BCELoss()
    loss = loss_func(sigmoid_output, target)

    return loss

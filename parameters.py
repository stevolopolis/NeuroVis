"""
This file contains the parameters for different sorts of AM
implementation in this repository.

This file is Copyright (c) 2022 Steven Tin Sui Luo.
"""

import torch
import os

class Params:
    """
    Parameters for visualization of model.
    """
    def __init__(self):
        # network name
        self.net = 'gr-convnet'
        self.MODEL_NAME = 'alexnetMap_grasp_top5_v3.2.2' #'alexnetMap_cls_top5_v2.3_epoch150'

        # device: cpu / gpu
        self.DEVICE = torch.device('cpu') if torch.cuda.is_available \
                                      else torch.device('cpu')

        # AM params
        self.OUTPUT_SIZE = 32  # 224/512 optimal for integrated-gradients # 32 for pixel-AM to fill input image
        self.N_CHANNELS = 4
        self.IMG_SIZE = (self.N_CHANNELS, self.OUTPUT_SIZE, self.OUTPUT_SIZE) 
        self.EPOCHS = 100
        self.LR = 0.1
        self.INIT_METHOD = 'noise'  # or 'zero'

        # FMAP params
        self.FMAP_LAYER = 'conv4'

        # Kernel params
        self.N_KERNELS = 64
        self.vis_layers = ['7']

        # Paths params
        self.VIS_PATH = 'vis/%s' % self.MODEL_NAME
        self.AM_PATH = 'kernel-am-grasp-pixel'
        self.ACT_PATH = 'neuro-activation'
        self.GRAD_PATH = 'saliency'
        self.AM_GRAD_PATH = 'guided-am'
        self.MODEL_PATH = 'trained-models'
        self.DATA_PATH = 'datasets/top_5/train'
        self.LABEL_FILE = 'cls_top_5.txt'

        # Visualization params
        self.N_IMG = 80
        self.vis_img_size = (224, 224)
        
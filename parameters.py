"""
This file contains the parameters for different sorts of AM
implementation in this repository.

This file is Copyright (c) 2022 Steven Tin Sui Luo.
"""

import torch
import os

from inference.models.alexnet import AlexnetMap_v5

class Params:
    """
    Parameters for visualization of model.
    """
    def __init__(self):
        # network name
        self.net = 'gr-convnet'
        self.MODEL_NAME = 'alexnetGrasp_cls_top5_v100_epoch100'

        # device: cpu / gpu
        self.DEVICE = torch.device('cpu') if torch.cuda.is_available \
                                      else torch.device('cpu')

        # AM params
        self.OUTPUT_SIZE = 224  # 224/512 optimal for integrated-gradients # 32 for pixel-AM to fill input image
        self.N_CHANNELS = 4
        self.IMG_SIZE = (self.N_CHANNELS, self.OUTPUT_SIZE, self.OUTPUT_SIZE) 
        self.EPOCHS = 100
        self.LR = 0.1
        self.INIT_METHOD = 'noise'  # or 'zero'

        # FMAP params
        self.FMAP_LAYER = 'conv4'

        # Kernel params
        self.N_KERNELS = 64
        self.vis_layers = ['0', '1', '2', '3', '4', '5', '6']

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
        self.N_IMG = 16
        self.vis_img_size = (128, 128)

        # selected model for visualization
        weights_path = os.path.join(self.MODEL_PATH, self.MODEL_NAME + '.pth')
        self.MODEL = AlexnetMap_v5(n_cls=5)
        self.MODEL.load_state_dict(torch.load(weights_path))
        
        self.MODEL.eval()
        
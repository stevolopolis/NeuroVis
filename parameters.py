"""
This file contains the parameters for different sorts of AM
implementation in this repository.

Parameters are mainly divided into different parts based on
the network under investigation:
    - gr-convnet
    - resnet18
    - vgg16

This file is Copyright (c) 2022 Steven Tin Sui Luo.
"""
import torch

class GrParams:
    """
    Parameters for AM visualization on gr-convnet
    """
    def __init__(self):
        # network name
        self.net = 'gr-convnet'

        # device: cpu / gpu
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available \
                                      else torch.device('cpu')
        # AM params
        self.IMG_SIZE = (4, 64, 64) 
        self.EPOCHS = 251
        self.LR = 1e-1
        self.INIT_METHOD = 'noise'  # or 'zero'

        # Kernel params
        self.N_KERNELS = 32
        self.INIT_METHOD = 'noise'
        self.vis_layers = ['conv1', 'conv2', 'conv3', 'res1', 'res2', 'res3']

        # Paths params
        self.PATH = 'kernel-am-grasp-pixel'
        self.MODEL_PATH = 'trained-models/epoch_19_iou_0.98'
        
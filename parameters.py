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
import os

from torchvision.models import alexnet

from inference.models.grconvnet_cls import GrCLS
from inference.models.alexnet import AlexNet, PretrainedAlexnet, myAlexNet, AlexnetMap_v5

class GrParams:
    """
    Parameters for AM visualization on gr-convnet
    """
    def __init__(self):
        # network name
        self.net = 'gr-convnet'
        self.MODEL_NAME = 'alexnetGrasp_grasp_top5_v110_epoch150'

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
        self.vis_layers = ['0', '2', '4', '6', '8']

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
        # Load gr-convnet model
        #self.MODEL = torch.load(weights_path, map_location=self.DEVICE)
        # grconvCLS
        #self.MODEL = GrCLS().to(self.DEVICE)
        # alexnetCLS
        #self.MODEL = AlexNet(input_channels=self.N_CHANNELS).to(self.DEVICE)
        #self.MODEL = PretrainedAlexnet().to(self.DEVICE)
        #self.MODEL = myAlexNet(input_channels=self.N_CHANNELS).to(self.DEVICE)
        self.MODEL = AlexnetMap_v5(n_cls=5)
        self.MODEL.load_state_dict(torch.load(weights_path))
        # alexnet imagenet
        #self.MODEL = alexnet(pretrained=True)
        
        self.MODEL.eval()
        
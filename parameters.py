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

class Params:
    def __init__(self):
        DEVICE = torch.device('cuda') if torch.cuda.is_available \
                                      else torch.device('cpu')
        

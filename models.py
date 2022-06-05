"""
This file contains a class that extracts a submodel from
a pretrained model, up until the selected layer for
visualization.
"""
import torch.nn as nn


class ExtractModel(nn.Module):
    """This class extracts a subset of a pretrained model
    up till the layer of visualization.
    
    There are three implementations for three different models:
        - resnet18
        - vgg16
        - gr-convnet
    """
    def __init__(self, model, layer, net_type='gr-convnet'):
        super().__init__()
        self.children_list = []

        # for VGG
        if net_type == 'vgg16':
            for n, c in model.features.named_children():
                self.children_list.append(c)
                if n == layer:
                    break
        # for gr-convnet
        elif net_type == 'gr-convnet':
            for n,c in model.named_children():
                self.children_list.append(c)
                if n[:2] == 'bn':
                    self.children_list.append(nn.ReLU())  # ****Very Important******
                if n == layer:
                    break
        elif net_type == 'gr-convnet-sin':
            prev_c = None
            for n,c in model.named_children():
                if n in ['pos_output', 'cos_output', 'width_output', 'dropout_cos', 'dropout_pos', 'dropout_wid']:
                    continue
                elif n == 'sin_output':
                    prev_c = c
                    continue
                self.children_list.append(c)
                if prev_c is not None:
                    self.children_list.append(prev_c)
                if n[:2] == 'bn':
                    self.children_list.append(nn.ReLU())  # ****Very Important******
        # for resnet18
        elif net_type == 'resnet18':
            for n, c in model.named_children():
                self.children_list.append(c)
                if n == layer:
                    break
        
        self.net = nn.Sequential(*self.children_list)
        
    def forward(self, x):
        x = self.net(x)
        return x


class ExtractOutputModel(nn.Module):
    """This class extracts a pretrained grasping model such
    that only a particular output in a set of outputs is returned.
    
    The current code is based on the architecture and indexing of 
    'gr-convnet'.
    """
    def __init__(self, model, output):
        super().__init__()
        self.net = model

        if output == 'width':
            self.idx = 3
        elif output == 'cos':
            self.idx = 1   
        elif output == 'sin':
            self.idx = 2
        elif output == 'quality':
            self.idx = 0

    def forward(self, x):
        return self.net(x)[self.idx]
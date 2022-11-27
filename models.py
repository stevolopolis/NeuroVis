"""
This file contains a class that extracts a submodel from
a pretrained model, up until the selected layer for
visualization.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExtractModel(nn.Module):
    """This class extracts a subset of a pretrained model
    up till the layer of visualization.
    
    There are three implementations for three different models:
        - resnet18
        - vgg16
        - gr-convnet
    """
    def __init__(self, model, layer, net_type='gr-convnet'):
        super(ExtractModel, self).__init__()
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


class ExtractAlexModel(nn.Module):
    def __init__(self, model, output, net_type=None):
        super(ExtractAlexModel, self).__init__()
        self.output = output
        self.layers = {}
        # ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1']
        for n, c in model.named_children():
            self.layers[n] = c

    def forward(self, x):
        for i in range(1, 5):
            conv = 'conv%s' % i
            bn = 'bn%s' % i
            x = self.layers[conv](x)
            x = self.layers[bn](x)
            if self.output == conv:
                return x
            x = F.relu(x)
            x = F.max_pool2d(x, 2, 2)

        x = self.layers['conv5'](x)
        if 'conv5' == self.output:
            return x
        x = self.layers['bn5'](x)
        x = F.relu(x)

        x=F.adaptive_avg_pool2d(x,1)
        x = x.reshape(x.size(0), -1)

        x = self.layers['fc1'](x)

        return x


class AlexnetModel(nn.Module):
    def __init__(self, model, output, net_type=None):
        super(AlexnetModel, self).__init__()
        self.output = output
        self.layers = {}
        for n, c in model.named_children():
            print(n, c)
            self.layers[n] = c

    def forward(self, x):
        for idx in self.layers:
            x = self.layers[idx](x)
            if idx == self.output:
                return x

        return x


class ExtractOutputModel(nn.Module):
    """This class extracts a pretrained grasping model such
    that only a particular output in a set of outputs is returned.
    
    The current code is based on the architecture and indexing of 
    'gr-convnet'.
    """
    def __init__(self, model, output):
        super(ExtractOutputModel, self).__init__()
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


# =================================================================
# AlexnetMap_v5 codes
# =================================================================
class AlexnetMapFeatures(nn.Module):
    """This class extracts the 'feature' module from AlexnetGrasp_v5 model.
        
    The indexing of the rgb / depth feature net is as follows:
        0 - nn.Conv2d(64+64, 32, kernel_size=5, padding=2),
        1 - nn.ReLU(inplace=True),
        2 - nn.Dropout(0.3),
        3 - nn.MaxPool2d(kernel_size=3, stride=2),
        4 - nn.Conv2d(32, 64, kernel_size=3, padding=1),
        5 - nn.ReLU(inplace=True),
        6 - vnn.Dropout(0.3),
        7 - nn.Conv2d(64, 64, kernel_size=3, padding=1),
        8 - nn.ReLU(inplace=True),
        9 - nn.Dropout(0.3),
        10 - nn.Conv2d(64, 64, kernel_size=3, padding=1),
        11 - nn.ReLU(inplace=True),
        12 - nn.Dropout(0.3),

    Specify the visualization layer using the above indexing pattern for
    <output> parameter.
    """
    def __init__(self, model, output):
        super(AlexnetMapFeatures, self).__init__()
        self.output = output
        self.model = model
        self.layers = {}
        for n, c in model.features.named_children():
            self.layers[n] = c

    def forward(self, x):
        rgb = x[:, :3, :, :]
        d = torch.unsqueeze(x[:, 3, :, :], dim=1)
        d = torch.cat((d, d, d), dim=1)

        rgb = self.model.rgb_features(rgb)
        d = self.model.d_features(d)
        x = torch.cat((rgb, d), dim=1)

        for idx in self.layers:
            x = self.layers[idx](x)
            if idx == self.output:
                return x

        return x


class AlexnetMapRgbdFeatures(nn.Module):
    """This class extracts the rgbd feature nets from AlexnetGrasp_v5 model.
    
    Users can specify whether to extract the 'rgb' or the 'depth' feature
    net to extract.
    
    The indexing of the rgb / depth feature net is as follows:
        0 - nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
        1 - nn.ReLU(inplace=True),
        2 - nn.MaxPool2d(kernel_size=3, stride=2),
        3 - nn.Conv2d(64, 192, kernel_size=5, padding=2),
        4 - nn.ReLU(inplace=True),
        5 - nn.MaxPool2d(kernel_size=3, stride=2)

    Specify the visualization layer using the above indexing pattern for
    <output> parameter.
    """
    def __init__(self, model, output, feature_type='rgb'):
        super(AlexnetMapRgbdFeatures, self).__init__()
        self.output = output
        self.feature_type = feature_type
        self.layers = {}
        if feature_type == 'rgb':
            for n, c in model.rgb_features.named_modules():
                if isinstance(c, nn.Conv2d):
                    self.layers[n] = c
        elif feature_type == 'd':
            for n, c in model.d_features.named_children():
                self.layers[n] = c

        print(self.layers)

    def forward(self, x):
        if self.feature_type == 'rgb':
            x = x[:, :3, :, :]
        elif self.feature_type == 'd':
            x = torch.unsqueeze(x[:, 3, :, :], dim=1)
            x = torch.cat((x, x, x), dim=1)

        for idx in self.layers:
            x = self.layers[idx](x)
            if idx == self.output:
                return x

        return x


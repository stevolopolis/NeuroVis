"""
Objectives of this file:
    1. Given a selected feature map, create the input image that maximizes the activations of that map
    2. Output the kernel image of the selected feature map
    3. Given a selected image, visualize the output of the selected feature map
    4. (optional) Visualize each individual feature map for comparison and do (1)-(3) for all
"""

import torch 
import cv2 as cv
from torchvision import transforms
from utils import create_rgbd_image, output_set_prep

from parameters import GrParams

params = GrParams()


class ActivationMaximization():
    """This class performs AM on a pretrained model.
    
    Users can specify the img_size, lr, and number of 
    iterations (epochs).
    
    Users can also specify the init_method:
        - noise -- initialize a guassian noise image for backprop
        - zero -- initialize a black image for backprop

    For torchvision models, a preprocessing pipeline is 
    also included.
    """
    def __init__(self, model, img_size, lr=1e-3, epochs=100,
                 init_method='noise', device='cuda'):
        self.model = model
        self.img_size = (1, img_size[0], img_size[1], img_size[2])

        # Initialize image with gaussian noise
        if params.INIT_METHOD == 'noise':
            # image initialization for gr-convnet (4 channels)
            if params.net == 'gr-convnet':
                self.backprop = torch.randn(self.img_size, dtype=torch.float32,
                                            requires_grad=True, device=device)
            # image initialization for gr-convnet (3 channels with normalization) 
            elif params.net in ('vgg16', 'resnet18'):
                self.backprop = torch.randint(150, 180, self.img_size,
                                              dtype=torch.float32, device=device)
                self.backprop = self.backprop / 255
                self.backprop.requires_grad = True        
        # Initialize image with black pixels
        elif init_method == 'zero':
            self.backprop = torch.full(self.img_size, 1e-4,
                                       dtype=torch.float32,
                                       requires_grad=True,
                                       device=device)

        # Copy start image for saving as jpg
        self.start = self.backprop.clone()
        self.lr = lr
        self.epochs = epochs
        self.loss = torch.tensor(0.0)

        # Preprocessing pipeline for torch.models
        # Not applicable for 'gr-convnet'
        self.preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.optim = torch.optim.Adam([self.backprop], lr=lr, weight_decay=1e-6)

    def backprop_full(self, kernel_idx, alpha=1, beta=1, sigma=1, p=1, save_img_set=False):
        pass

    def backprop_pixel(self, kernel_idx, alpha=1, beta=1, sigma=1, p=1, save_img_set=False):
        for _ in range(self.epochs):
            self.optim.zero_grad()

            # for vgg / resnet
            if params.net in ('vgg16', 'resnet18'):
                input = self.preprocess(self.backprop_img.squeeze()).unsqueeze(0)
            # for gr-convnet
            elif params.net == 'gr-convnet':
                input = self.backprop_img

            self.output = self.model(input)
            self.fmap_output = self.output[0, kernel_idx]
            
            # Getting center coordinate of feature map
            h = self.fmap_output.shape[0]
            w = self.fmap_output.shape[1]
            x_mid = w // 2
            y_mid = h // 2

            self.pixel_loss -= torch.mean(self.fmap_output[y_mid, x_mid])
            self.loss = self.pixel_loss

            self.loss.backward()
            self.optim.step()

        # Target feature map with maxed out pixel values
        self.target = torch.ones(self.selected_output.shape)
        return self.show_loss(self.epochs)

    def show_loss(self):
        rounded_loss = round(self.loss.item(), 5)
        print('Epoch: %s\t Loss: %s' % (self.epochs, rounded_loss))
        start_img, backprop_img, target_img, fmap_img = output_set_prep(self.start,
                                                                        self.backprop,
                                                                        self.target,
                                                                        self.fmap_output)
        # for vgg/resnet
        backprop_img = create_rgbd_image(backprop_img, normalize=True)
        # for gr-convnet
        start_img = create_rgbd_image(start_img, normalize=True)
        backprop_img = create_rgbd_image(backprop_img, normalize=True)
        target_img = create_rgbd_image(target_img, normalize=True)
        fmap_img = create_rgbd_image(fmap_img, normalize=True)
        
        return start_img, backprop_img, target_img, fmap_img
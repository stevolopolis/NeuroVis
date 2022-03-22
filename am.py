"""
Objectives of this file:
    1. Given a selected feature map, create the input image that maximizes the activations of that map
    2. Output the kernel image of the selected feature map
    3. Given a selected image, visualize the output of the selected feature map
    4. (optional) Visualize each individual feature map for comparison and do (1)-(3) for all
"""

import torch 
import math
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
from scipy.ndimage.interpolation import rotate
from scipy.ndimage import gaussian_filter
from torchvision import transforms
from utils.my_utils import create_rgbd_image, output_set_prep


class ActivationMaximization():
    """"""
    def __init__(self, model, img_size, lr=1e-3, epochs=100, device='cuda',
                 fill_method='rect', init_method='noise', max_feature='width',
                 width_boundary_ratio=0.1, angle=0, save_path='rect', loss_update_freq=10):
        """
        self.img_size: (img_w, img_h, img_c)
        self.activation_index: [<layer_str>, <channel_idx>]
        """
        self.model = model.to(device)
        self.img_size = (1, img_size[0], img_size[1], img_size[2])
        self.device = device
        self.fill_method = fill_method
        self.init_method = init_method
        self.max_feature = max_feature
        self.width_boundary_ratio = width_boundary_ratio
        self.angle = angle
        self.directory = save_path
        self.loss_update_freq = loss_update_freq
        if init_method == 'noise':
            # for vgg/resnet
            self.backprop_img = torch.randint(150, 180, self.img_size, dtype=torch.float32, device=device)
            self.backprop_img = self.backprop_img / 255
            self.backprop_img.requires_grad = True
            # for gr-convnet
            # self.backprop_img = torch.randn(self.img_size, dtype=torch.float32, requires_grad=True, device=device)
        elif init_method == 'zero':
            self.backprop_img = torch.full(self.img_size, 1e-4, dtype=torch.float32, requires_grad=True, device=device)
        self.start_img = self.backprop_img.clone()
        self.lr = lr
        self.epochs = epochs

        self.preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.optim = torch.optim.Adam([self.backprop_img], lr=lr, weight_decay=1e-6)
        self.criterion = nn.BCELoss()

    def activation_maximization(self, kernel_idx_set, alpha=1, beta=1, sigma=1, p=1, save_img_set=False):
        if type(kernel_idx_set) == int:
            kernel_idx_set = [kernel_idx_set]
            
        for epoch in range(self.epochs):
            self.optim.zero_grad()

            self.pixel_loss = 0
            
            # for vgg / resnet
            input = self.preprocess(self.backprop_img.squeeze()).unsqueeze(0)
            # for gr-convnet
            # input = self.backprop_img

            self.output = self.model(input)
            # Adding jitter and guassian blur
            #trans_backprop_img = translate_tensor(input)
            #trans_backprop_img = gaussian_blur(self.backprop_img, device=self.device)
            #self.output_jitter = self.model(trans_backprop_img)
            for kernel_idx in kernel_idx_set:
                self.selected_output = self.output[0, kernel_idx]

                #####
                # Pixel AM implementation
                #####
                h = self.selected_output.shape[0]
                w = self.selected_output.shape[1]
                x_mid = w // 2
                y_mid = h // 2

                self.pixel_loss -= torch.mean(self.selected_output[y_mid, x_mid]) # torch.sum(self.selected_output[:, 28]) # [28, :]  # torch.sum also works (enormous loss value with greater AM clarity)
                #self.selected_jitter_output = self.output_jitter[0][kernel_idx]
                #self.pixel_loss -= torch.mean(self.selected_jitter_output)
                #target = torch.zeros_like(self.output_jitter[0])
                #target[kernel_idx] = 1
                #target.to(self.device)
                #self.pixel_loss = self.criterion(F.softmax(self.output_jitter[0], dim=0), target)

            self.tv_loss = TVLoss(self.backprop_img)
            self.lp_loss = LPLoss(self.backprop_img, p=p)
            loss = self.pixel_loss * alpha + self.lp_loss * beta + self.tv_loss * sigma
            loss.backward()
            self.optim.step()

            # Scratch
            self.max_target = torch.ones(self.selected_output.shape)

            self.show_loss(self.epochs, epoch, loss, save_img_set=save_img_set, output_max=False)
        
        return self.start_img, self.backprop_img

    def pixel_maximization(self, kernel_idx_set, alpha=1, sigma=1, save_img_set=False):
        if type(kernel_idx_set) == int:
            kernel_idx_set = [kernel_idx_set]
            
        for epoch in range(self.epochs):
            self.optim.zero_grad()

            self.pixel_loss = 0
            self.output = self.model(self.backprop_img)
            for kernel_idx in kernel_idx_set:
                self.selected_output = self.output[0][kernel_idx]
                self.pixel_loss -= self.selected_output[28][28] # torch.sum(self.selected_output[:, 28]) # [28, :]  # torch.sum also works (enormous loss value with greater AM clarity)

            self.tv_loss = TVLoss(self.backprop_img)
            loss = self.pixel_loss * alpha + self.tv_loss * sigma
            loss.backward()
            self.optim.step()

            self.max_target = torch.zeros(self.selected_output.shape)
            self.max_target[28][28] = 1
            self.show_loss(self.epochs, epoch, loss, save_img_set=save_img_set, output_max=False)

        return self.start_img, self.backprop_img

    def output_maximization(self, output_idx, save_img_set=False):
        for epoch in range(self.epochs):
            self.optim.zero_grad()

            output = self.model(self.backprop_img)
            self.selected_output = output[output_idx]
            if epoch == 0:
                self.synth_selection()
            loss = self.criterion(self.selected_output, self.max_target)
            loss.backward()
            self.optim.step()

            self.show_loss(self.epochs, epoch, loss, save_img_set=save_img_set, output_max=True)

        return self.start_img, self.backprop_img, self.max_target, self.selected_output

    def synth_selection(self):
        target_value = self.get_target_value()
        self.width_synth(target_value)
        self.angle_synth()

    def width_synth(self, target_value):
        if self.fill_method == 'rect':
            self.rect_fill(target_value)
        elif self.fill_method == 'circle':
            self.circle_fill(target_value)

    def rect_fill(self, target_value):
        """Width output range = [0, max(self.img_size[2], self.img_size[3])]"""
        boundary_size = int(self.img_size[2] * self.width_boundary_ratio)
        self.max_target = torch.zeros(self.selected_output.shape, dtype=torch.float32).to(self.device)
        left_bound = int(self.img_size[2] / 2) - boundary_size
        right_bound = int(self.img_size[2] / 2) + boundary_size
        upper_bound = int(self.img_size[3] /2) - boundary_size*2
        lower_bound = int(self.img_size[3] / 2) + boundary_size*2
        self.max_target[:, :, upper_bound:lower_bound, left_bound:right_bound] = target_value

    def circle_fill(self, target_value):
        radius = self.img_size[2] * self.width_boundary_ratio / 2
        mid_x = int(self.img_size[3] * 0.5)
        mid_y = int(self.img_size[2] * 0.5)
        self.max_target = torch.zeros(self.selected_output.shape, dtype=torch.float32).to(self.device)
        for x_coord in range(self.img_size[3]):
            for y_coord in range(self.img_size[2]):
                dist = math.sqrt((mid_x - x_coord) ** 2 + (mid_y - y_coord) ** 2)
                if dist < radius:
                    self.max_target[:, :, y_coord, x_coord] = target_value

    def angle_synth(self):
        """
        Temporary solution using numpy arrays.
        Optimal solution is to use tensors,
        or rotate in previous sections of the code to improve efficiency and readability.
        """
        max_target_arr = self.max_target.detach().cpu().numpy()
        max_target_arr = rotate(max_target_arr, angle=self.angle, axes=(2, 3), reshape=False, mode='constant', cval=0.0)
        self.max_target = torch.tensor(max_target_arr).to(self.device)

    def get_target_value(self):
        if self.max_feature == 'width':
            return max(self.img_size[2], self.img_size[3])
        elif self.max_feature == 'cos':
            return math.cos(2 * (self.angle / 180 * math.pi))
        elif self.max_feature == 'sin':
            return math.sin(2 * (self.angle / 180 * math.pi))
        else:
            return 1

    def show_kernel_map(self):
        pass

    def show_single_kernel_output(self):
        pass

    def show_loss(self, epochs, epoch, loss, save_img_set=False, output_max=True):
        if epoch % self.loss_update_freq == 0:
            rounded_loss = round(loss.item(), 5)
            print('Epoch: %s\t Loss: %s' % (epoch, rounded_loss))
            if save_img_set:
                save_start_img, save_backprop_img, save_max_target, save_selected_out = output_set_prep(self.start_img,
                                                                                                        self.backprop_img,
                                                                                                        self.max_target,
                                                                                                        self.selected_output)
                # for vgg/resnet
                save_backprop_rgb = create_rgbd_image(save_backprop_img, normalize=True)
                # for gr-convnet
                #save_backprop_rgb, save_backprop_d = create_rgbd_image(save_backprop_img, normalize=True)
                #save_max_target = create_rgb_image(save_max_target, normalize=True)
                save_selected_out = create_rgbd_image(save_selected_out, normalize=True)
                
                if output_max:
                    img_set_dir = '%s-%s-%s-%s-%s-%s' % (self.fill_method, self.max_feature, self.width_boundary_ratio, self.angle, self.init_method, epochs)
                    img_set_str = '%s-%s-%s-%s-%s-%s' % (self.fill_method, self.max_feature, self.width_boundary_ratio, self.angle, self.init_method, epoch)
                else:
                    img_set_dir = '%s-%s-%s' % (self.max_feature, self.init_method, epochs)
                    img_set_str = '%s-%s-%s' % (self.max_feature, self.init_method, epoch)

                #cv.imwrite('vis/%s/%s/%s_end_img_d.png' % (self.directory, img_set_dir, img_set_str), save_backprop_d)
                cv.imwrite('vis/%s/%s/%s_end_img_rgb.png' % (self.directory, img_set_dir, img_set_str), save_backprop_rgb)
                #cv.imwrite('vis/%s/%s/%s_target_img.png' % (self.directory, img_set_dir, img_set_str), save_max_target)
                cv.imwrite('vis/%s/%s/%s_output_img.png' % (self.directory, img_set_dir, img_set_str), save_selected_out)


def gaussian_blur(input, device):
    temp = input.squeeze(0)
    temp = temp.cpu().detach().numpy()
    for channel in range(3):
        cimg = gaussian_filter(temp[channel], 1)
        temp[channel] = cimg
    temp = torch.from_numpy(temp).to(device)
    input = temp.unsqueeze(0)

    return input


def get_rand_translation():
    trans_x = random.randint(0, 0)
    trans_y = random.randint(0, 0)
    return trans_x, trans_y


def translate_tensor(tensor):
    trans_x, trans_y = get_rand_translation()

    return tensor[:, :, trans_y:, trans_x:]

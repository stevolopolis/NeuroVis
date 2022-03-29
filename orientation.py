"""
This file contains the <OrientationTest> class which 
creates a variety of synthetic objects of different 
orientation that gets fed into a pretrained network.

Then, the feature maps of selected layers and kernels are
visualized for future testing, which may include ranking
metrics that compare the orientation specializations of
each kernel.
"""

import torch 
import cv2 as cv
from torchvision import transforms

from parameters import GrParams

params = GrParams()


class OrientationTest():
    """This class performs AM on a pretrained model.
    
    Users can specify the img_size, lr, and number of 
    iterations (epochs).
    
    Users can also specify the init_method:
        - noise -- initialize a guassian noise image for backprop
        - zero -- initialize a black image for backprop

    For torchvision models, a preprocessing pipeline is 
    also included.
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device

        # Preprocessing pipeline for torch.models
        # Not applicable for 'gr-convnet'
        self.preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.optim = torch.optim.Adam([self.backprop], lr=lr, weight_decay=1e-6)

    def run(self):
        pass

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

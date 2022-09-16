"""
This file contains the Path class that creates, deletes, or
modifies paths within the directory for saving AM results and
other related images.

This file is Copyright (c) 2022 Steven Tin Sui Luo.
"""

import os
import shutil
import glob

from parameters import GrParams

params = GrParams()

class GrPath:
    """This class prepares the directories for saving AM results
    in the <vis> folder.
    """
    def __init__(self, net: str):
        self.net = net
        self.model_name = params.MODEL_NAME
        self.vis_path = params.VIS_PATH
        self.am_path = params.AM_PATH
        self.act_path = params.ACT_PATH
        self.grad_path = params.GRAD_PATH
        self.guided_am_path = params.AM_GRAD_PATH

        # self.main_path determines the path of concern in the current use case
        self.main_path = self.am_path

        # Create <vis> folder to save all visualizations.
        if 'vis' not in os.listdir('./'):
            os.makedirs('vis')
        if self.model_name not in os.listdir('vis'):
            os.makedirs(self.vis_path)

    def create_am_path(self):
        """This method creates a subdirectory in <vis> for am visualization."""
        if self.am_path not in os.listdir(self.vis_path):
            os.makedirs(os.path.join(self.vis_path, self.am_path))

        # Swith main operating path to self.am_path
        self.main_path = self.am_path
        self.save_subdir = os.path.join(self.vis_path, self.main_path)

    def create_act_path(self):
        """This method creates a subdirectory in <vis> for neuron activation
        visualization.
        
        Activation Maps includes those generated from real images and synthetic images
        (e.g. synthetic rectangular boxes).
        """
        if self.act_path not in os.listdir(self.vis_path):
            os.makedirs(os.path.join(self.vis_path, self.act_path))

        # Swith main operating path to self.act_path
        self.main_path = self.act_path
        self.save_subdir = os.path.join(self.vis_path, self.main_path)

    def create_grad_path(self):
        """This method creates a subdirectory in <vis> for saliency maps.

        Saliency maps are generated from vanila gradients or integrated
        gradients. There is also the option to multiple gradients to the 
        values of the image itself for 'better clarity'.
        """
        if self.grad_path not in os.listdir(self.vis_path):
            os.makedirs(os.path.join(self.vis_path, self.grad_path))

        # Swith main operating path to self.act_path
        self.main_path = self.grad_path
        self.save_subdir = os.path.join(self.vis_path, self.main_path)
    
    def create_guided_am_path(self):
        """This method creates a subdirectory in <vis> for guided AM."""
        if self.guided_am_path not in os.listdir(self.vis_path):
            os.makedirs(os.path.join(self.vis_path, self.guided_am_path))

        # Swith main operating path to self.act_path
        self.main_path = self.guided_am_path
        self.save_subdir = os.path.join(self.vis_path, self.main_path)
    
    def create_layer_paths(self, layer: str):
        """This method creates sub-directory in <self.am_path> for a specific layer."""
        if layer not in os.listdir(os.path.join(self.vis_path, self.main_path)):
            os.makedirs(os.path.join(self.vis_path, self.main_path, layer))

        self.save_subdir = os.path.join(self.vis_path, self.main_path, layer)

    def create_kernel_paths(self, layer:str, kernel: str):
        """This method creates sub-directory in <self.am_path> for a specific layer."""
        if kernel not in os.listdir(os.path.join(self.vis_path, self.main_path, layer)):
            os.makedirs(os.path.join(self.vis_path, self.main_path, layer, kernel))

        self.save_subdir = os.path.join(self.vis_path, self.main_path, layer, kernel)

    def create_img_paths(self, img_id: str):
        """This method creates sub-directory in <self.guided_am_path> for a specific image."""
        curr_path = os.path.join(self.save_subdir, img_id)
        if img_id not in os.listdir(self.save_subdir):
            os.makedirs(os.path.join(self.save_subdir, img_id))
            #original_img_path = find_image(img_id)
            #shutil.copyfile(original_img_path, os.path.join(curr_path, 'image.png'))

        self.save_subdir = curr_path


def find_image(img_id):
    for file_path in glob.glob('%s/*/*/*RGB.png' % params.DATA_PATH):
        img_name = file_path.split('\\')[-1][:-8]
        if img_id == img_name:
            return file_path

    
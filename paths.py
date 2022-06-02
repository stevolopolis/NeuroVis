"""
This file contains the Path class that creates, deletes, or
modifies paths within the directory for saving AM results and
other related images.

This file is Copyright (c) 2022 Steven Tin Sui Luo.
"""

import os

from parameters import GrParams

params = GrParams()

class GrPath:
    """This class prepares the directories for saving AM results
    in the <vis> folder.
    """
    def __init__(self, net: str):
        self.net = net
        self.am_path = params.AM_PATH
        self.act_path = params.ACT_PATH

        # self.main_path determines the path of concern in the current use case
        self.main_path = self.am_path

        # Create <vis> folder to save all visualizations.
        if 'vis' not in os.listdir('./'):
            os.makedirs('vis')

    def create_am_path(self):
        """This method creates a subdirectory in <vis> for am visualization."""
        if self.am_path not in os.listdir('vis'):
            os.makedirs(os.path.join('vis', self.am_path))

        # Swith main operating path to self.am_path
        self.main_path = self.am_path

    def create_act_path(self):
        """This method creates a subdirectory in <vis> for neuron activation
        visualization.
        
        Activation Maps includes those generated from real images and synthetic images
        (e.g. synthetic rectangular boxes).
        """
        if self.act_path not in os.listdir('vis'):
            os.makedirs(os.path.join('vis', self.act_path))

        # Swith main operating path to self.act_path
        self.main_path = self.act_path
    
    def create_layer_paths(self, layer: str):
        """This method creates sub-directory in <self.am_path> for a specific layer."""
        if layer not in os.listdir(os.path.join('vis', self.main_path)):
            os.makedirs(os.path.join('vis', self.main_path, layer))

        self.save_subdir = os.path.join('vis', self.main_path, layer)

    
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
        self.main_path = params.PATH

        # Create directory for currect experiment, if not already existing.
        if self.main_path not in os.listdir('vis'):
            os.makedirs(os.path.join('vis', self.main_path))
    
    def create_layer_paths(self, layer: str):
        """This method creates sub-directory in <vis> for a specific layer."""
        if layer not in os.listdir(os.path.join('vis', self.main_path)):
            os.makedirs(os.path.join('vis', self.main_path, layer))

        self.save_subdir = os.path.join('vis', self.main_path, layer)
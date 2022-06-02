import glob
import torch
import os
import numpy as np

from PIL import Image
from parameters import GrParams

# Params class containing parameters for all visualization.
params = GrParams()

class DataLoader:
    def __init__(self):
        pass
    
    def load_rgbd(self):
        """Returns one-by-one all RGB images in the <dataset> folder."""
        for img_path in glob.glob('%s/*/*RGB.png' % params.DATA_PATH):
            # Get image subdirectory name ("left" / "right")
            img_dir_str = img_path.split('\\')[-2]
            # Get image id (e.g. 0_4e4a043d8c8cee6afad30cd586639ed2)
            img_path_str = img_path.split('\\')[-1]
            img_id = img_path_str[:-7]
            # Open RGB image with PIL
            img_rgb = np.array(Image.open(img_path))
            # Open depth image with PIL
            img_d = np.array(Image.open(os.path.join(params.DATA_PATH, img_dir_str, img_id + 'mask.png')))

            yield (self.process(img_rgb, img_d), img_id)

    def process(self, rgb, d):
        """Returns rgbd image with correct format for inputing to model."""
        d = np.expand_dims(d, 2)
        img = np.concatenate((rgb, d), axis=2)
        img = np.moveaxis(img, -1, 0)
        img = np.expand_dims(img, 0)
        img = torch.tensor(img, dtype=torch.float32).to(params.DEVICE)

        return img
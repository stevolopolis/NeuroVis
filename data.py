import glob
import torch
import os
import numpy as np

from PIL import Image
from parameters import GrParams
from roboticGrasp.dataset_processing import grasp

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
            img_id = img_path_str[:-8]
            # Open RGB image with PIL
            img_rgb = Image.open(img_path)
            img_rgb = img_rgb.resize((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
            img_rgb = np.array(img_rgb)
            # Open depth image with PIL
            img_d = Image.open(os.path.join(params.DATA_PATH, img_dir_str, img_id + '_mask.png'))
            img_d = img_d.resize((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
            img_d = np.array(img_d)

            yield (self.process(img_rgb, img_d), img_id)
    
    def process(self, rgb, d):
        """Returns rgbd image with correct format for inputing to model."""
        d = np.expand_dims(d, 2)
        img = np.concatenate((rgb, d), axis=2)
        img = np.moveaxis(img, -1, 0)
        img = np.expand_dims(img, 0)
        img = torch.tensor(img, dtype=torch.float32).to(params.DEVICE)

        return img


def jacquard_sin_loader(img_id):
    """Returns 'sin' ground truth map for images take from the Jaquard Dataset.
    
    Code referenced from @author: Sulabh Kumra - https://github.com/skumra/robotic-grasping"""
    # Get ground-truth path from img_id
    path = glob.glob('%s/*/%s_grasps.txt' % (params.DATA_PATH, img_id))[0]
    # Load all grasp rectangles from .txt file
    bbs = grasp.GraspRectangles.load_from_jacquard_file(path, scale=params.OUTPUT_SIZE / 1024.0)
    # Convert grasp rectangles into one single 'cos' map
    _, ang_img, _ = bbs.draw((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
    sin = numpy_to_torch(np.sin(2 * ang_img))

    return sin


def numpy_to_torch(s):
    if len(s.shape) == 2:
        return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
    else:
        return torch.from_numpy(s.astype(np.float32))
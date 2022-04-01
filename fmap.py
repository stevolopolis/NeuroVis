"""

"""

import torch
import glob
import os
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from parameters import GrParams
from paths import GrPath
from utils import get_real_img_from_path, tensor2array, array2img, fmap_img_mat
from models import ExtractModel

# Params class containing parameters for AM visualization.
params = GrParams()
# Path class for managing required directories
# gr-convnet / resnet18 / vgg16
paths = GrPath('gr-convnet')
paths.create_fmap_path()

# Load trained gr-convnet model
model = torch.load(params.MODEL_PATH, map_location=params.DEVICE)
model.eval()

for i, img_path in enumerate(glob.glob('%s/*/*RGB.png' % params.DATA_PATH)):
    img, id = get_real_img_from_path(img_path)
    
    img_set = []
    for kernel_idx in tqdm(range(128)):
        # Create submodel with output = selected kernel
        ext_model = ExtractModel(model, params.FMAP_LAYER, net_type=params.net, device=params.DEVICE)

        # Generate feature map
        fmap = ext_model(img)[0][kernel_idx]

        # Convert feature map tensor to visualizable image and save
        fmap_arr = tensor2array(fmap.repeat(1, 3, 1, 1))
        fmap_img = array2img(fmap_arr)
        img_set.append(fmap_img)
    
    paths.create_layer_paths(params.FMAP_LAYER)
    save_img_path = '%s/fmap-matrix-%s.png' % (paths.save_subdir, id)
    fmap_mat = fmap_img_mat(img_set)
    cv2.imwrite(save_img_path, fmap_mat)

"""
This code visualizes the neuron activation maps of specified kernels
in a CNN, which aids our understanding of the purposes of each
individual neuron in the neural network.

The activation maps are generated from the Cornell Dataset.
The images are saved in the <datasets> folder.

This file is Copyright (c) 2022 Steven Tin Sui Luo.
"""
import torch
import cv2

from parameters import GrParams
from paths import GrPath
from models import ExtractModel
from data import DataLoader
from utils import tensor2array, array2img

# Params class containing parameters for all visualization.
params = GrParams()
# Path class for managing required directories
# gr-convnet / resnet18 / vgg16
paths = GrPath('gr-convnet')
paths.create_act_path()

# Load trained gr-convnet model
model = torch.load(params.MODEL_PATH, map_location=params.DEVICE)
model.eval()

# DataLoader for Cornell Dataset
dataLoader = DataLoader()

# AM visualization
for vis_layer in params.vis_layers:
    print('Visualizing for %s layer' % vis_layer)

    # AM on individual kernels
    for kernel_idx in range(params.N_KERNELS):
        print('Visualizing Kernel INDEX: %s' % kernel_idx)
        # Create submodel with output = selected kernel
        ext_model = ExtractModel(model, vis_layer, net_type=params.net, device=params.DEVICE)
        # Load Cornell Dataset images
        for img, img_id in dataLoader.load_rgbd():
            save_img_path = '%s/%s_%s_%s.png' % (paths.save_subdir, img_id, vis_layer, kernel_idx)
            # Feed image model and get act. map
            output = ext_model(img)
            act_map = output[0, kernel_idx]
            # Save act. map
            act_map_arr = tensor2array(act_map)
            act_img = array2img(act_map_arr)
            cv2.imwrite(save_img_path, act_img)
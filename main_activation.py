"""
This code visualizes the neuron activation maps of specified kernels
in a CNN, which aids our understanding of the purposes of each
individual neuron in the neural network.

The activation maps are generated from the Cornell Dataset.
The images are saved in the <datasets> folder.

This file is Copyright (c) 2022 Steven Tin Sui Luo.
"""
import cv2
import os

from parameters import Params
from paths import Path
from models import AlexnetMapRgbdFeatures, AlexnetMapFeatures
from data_v2 import DataLoaderV2
from utils import tensor2array, array2img, tensor2img, get_layer_width

# Params class containing parameters for all visualization.
params = Params()
# Path class for managing required directories
paths = Path()
paths.create_act_path()

# Load trained gr-convnet model
model = params.MODEL

# Select data for visualization
data = []
# DataLoader
dataLoader = DataLoaderV2('datasets/top_5_compressed_paperspace/test', 1)
for i, datum in enumerate(dataLoader.load_grasp()):
    if i >= params.N_IMG:
        break
    data.append(datum)

# Generating neuro activations (feature maps)
for vis_layer in params.vis_layers:
    print('Visualizing for %s layer' % vis_layer)

    # Create sub-directory for chosen layer
    paths.create_layer_paths(vis_layer)

    # Create submodel with output = selected kernel
    # AlexnetGrasp_v5
    ext_model = AlexnetMapRgbdFeatures(model, vis_layer, feature_type='rgb')
    #ext_model = AlexnetMapFeatures(model, vis_layer)
    n_kernels = get_layer_width(ext_model)

    for kernel_idx in range(n_kernels):
        print('Visualizing Kernel INDEX: %s' % kernel_idx)
        
        # Create sub-directory for chosen kernel
        paths.create_kernel_paths(vis_layer, str(kernel_idx))

        # AlexnetGrasp_v5
        for img, map, img_id in data:
            save_img_path = '%s/%s_%s_%s_neuro_activation.png' % (paths.save_subdir, img_id, vis_layer, kernel_idx)
            # Feed image model and get act. map
            
            for vis_layer in range(20):
                #ext_model = AlexnetMapRgbdFeatures(model, vis_layer, feature_type='rgb')
                ext_model = AlexnetMapFeatures(model, str(vis_layer))
                output = ext_model(img)

            act_map = output[0, kernel_idx]
            # Save act. map
            act_map_arr = tensor2array(act_map)
            act_img = array2img(act_map_arr, cv2.INTER_LINEAR, (64, 64))
            cv2.imwrite(save_img_path, act_img)
            # Save original image
            save_img_rgb, save_img_d = tensor2img(img)
            if '%s_image.png' % img_id not in os.listdir(os.path.join(paths.vis_path, paths.main_path)):
                cv2.imwrite(os.path.join(paths.vis_path, paths.main_path, '%s_image.png' % img_id), save_img_rgb)

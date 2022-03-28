"""
This code applies activation maximization (AM) on CNN kernels
to visualize the specialization of each kernel.

Available AM losses and regularizers include:
    - mean-of-feature
    - mean-of-feature-pixel
    - TV
    - Jitter
    - LP

This file is Copyright (c) 2022 Steven Tin Sui Luo.
"""
import torch
import cv2

from parameters import GrParams
from paths import GrPath
from models import ExtractModel
from am import ActivationMaximization
from utils import am_img_mat

# Params class containing parameters for AM visualization.
params = GrParams()
# Path class for managing required directories
# gr-convnet / resnet18 / vgg16
paths = GrPath('gr-convnet')

# AM visualization
for vis_layer in params.vis_layers:
    print('Visualizing for %s layer' % vis_layer)
    # Create sub-directory for chosen layer
    paths.create_layer_paths(vis_layer)

    # AM on individual kernels
    for kernel_idx in range(params.N_KERNELS):
        save_img_path = '%s/%s_%s.png' % (paths.save_subdir, vis_layer, kernel_idx)

        # Load trained gr-convnet model
        model = torch.load(params.MODEL_PATH, map_location=params.DEVICE)
        model.eval()

        # Create submodel with output = selected kernel
        ext_model = ExtractModel(model, vis_layer, net_type=params.net, device=params.DEVICE)
        am_func = ActivationMaximization(ext_model, params.IMG_SIZE, params.LR,
                                         params.EPOCHS, params.INIT_METHOD, params.DEVICE)
        # Run Activation Maximization
        print('Running AM on layer %s kernel %s' % (vis_layer, kernel_idx))
        am_func.backprop_pixel(kernel_idx)

        # Visualize/Save AM results
        am_img_set = am_func.show_am()
        img_matrix = am_img_mat(am_img_set)
        cv2.imwrite(save_img_path, img_matrix)

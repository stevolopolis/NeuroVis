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
import cv2

from parameters import Params
from paths import Path
from models import AlexnetMapRgbdFeatures, AlexnetMapFeatures
from am import ActivationMaximization
from utils import am_img_mat, get_layer_width

# Params class containing parameters for AM visualization.
params = Params()
# Path class for managing required directories
# gr-convnet / resnet18 / vgg16
paths = Path()
paths.create_am_path()

# Trained model paths
model = params.MODEL

# AM visualization
for vis_layer in params.vis_layers:
    print('Visualizing for %s layer' % vis_layer)
    # Create sub-directory for chosen layer
    paths.create_layer_paths(vis_layer)

    # Create submodel with output = selected kernel
    ext_model = AlexnetMapRgbdFeatures(model, vis_layer, feature_type='rgb')
    #ext_model = AlexnetMapFeatures(model, vis_layer)
    n_kernels = get_layer_width(ext_model)

    # AM on individual kernels
    for kernel_idx in range(n_kernels):
        save_img_path = '%s/%s_%s.png' % (paths.save_subdir, vis_layer, kernel_idx)

        #ext_model = model
        am_func = ActivationMaximization(ext_model, params.IMG_SIZE, params.LR,
                                         params.EPOCHS, params.INIT_METHOD, params.DEVICE)
        # Run Activation Maximization
        print('Running AM on layer %s kernel %s' % (vis_layer, kernel_idx))
        pixel_result_set, full_result_set = am_func.am(kernel_idx)
        #start_img, backprop_img = am_func.backprop_pred(kernel_idx)

        # Visualize/Save AM results
        img_matrix = am_img_mat(pixel_result_set, full_result_set)
        cv2.imwrite(save_img_path, img_matrix)
        #img_set = am_func.process_results(start_img, backprop_img, start_img, backprop_img)
        #cv2.imwrite(save_img_path, img_set[2])
        

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
import random
import torch

from parameters import Params
from paths import Path
import models
from inference.models.alexnet import AlexnetMap_v3
from inference.models.alexnet_ductranvan import Net
from am import ActivationMaximization
from utils import am_img_mat, get_layer_width

# Params class containing parameters for AM visualization.
params = Params()
# Path class for managing required directories
# gr-convnet / resnet18 / vgg16
paths = Path()
paths.create_am_path()

# Initialized pretrained alexnetMap model
model = AlexnetMap_v3().to(params.DEVICE)
model.load_state_dict(torch.load('trained-models/%s/%s_epoch150.pth' % (params.MODEL_NAME, params.MODEL_NAME)))
model.eval()

"""model = Net({"input_shape": (3,256,256),
            "initial_filters": 16, 
            "num_outputs": 5}).to(params.DEVICE)
model.load_state_dict(torch.load('trained-models/weights.pt'))
model.eval()"""

# AM visualization
for vis_layer in params.vis_layers:
    print('Visualizing for %s layer' % vis_layer)
    input()
    # Create sub-directory for chosen layer
    paths.create_layer_paths(vis_layer)

    # Create submodel with output = selected kernel
    #ext_model = models.AlexnetMapRgbdFeatures(model, vis_layer, feature_type='rgb')
    ext_model = models.AlexnetMapFeatures(model, vis_layer)

    #ext_model = model
    n_kernels = get_layer_width(ext_model)

    # AM on individual kernels
    kernels = [i for i in range(n_kernels)]
    random.shuffle(kernels)
    for kernel_idx in kernels:
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
        

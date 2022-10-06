"""
This code visualizes the saliency map of a CNN using integrated gradients,
which highlights which areas in an image influences the model prediction
the most. 

For classification models, we take the class label as the ground
truth for calculating gradients. For grasping models, we take the 
'cos' output of the final layer as the ground truth. Other outputs
such as the 'width', 'sin', etc. could also be used if required.

The saliency maps are generated from the Jacquard Dataset.
The images are saved in the <datasets> folder.

This file has referenced codes from @author: Utku Ozbulak - github.com/utkuozbulak.

This file is Copyright (c) 2022 Steven Tin Sui Luo.
"""

import os
import cv2

from parameters import Params
from paths import Path

from data_v2 import DataLoaderV2
from models import AlexnetMapRgbdFeatures, AlexnetMapFeatures
from guided_backprop import GuidedBackprop

from utils import tensor2img, get_layer_width
from misc_functions import (convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)

# Params class containing parameters for AM visualization.
params = Params()
# Path class for managing required directories
paths = Path()
paths.create_guided_am_path()

# Select data for visualization
data = []
# DataLoader
dataLoader = DataLoaderV2('datasets/top_5_compressed_paperspace/test', 1)
for i, datum in enumerate(dataLoader.load_grasp()):
    if i >= params.N_IMG:
        break
    data.append(datum)

# Trained model paths
model = params.MODEL
for vis_layer in params.vis_layers:
    print('Visualizing for %s layer' % vis_layer)
    # Create sub-directory for chosen layer
    paths.create_layer_paths(vis_layer)
    
    # Create submodel with output = selected kernel
    #ext_model = AlexnetMapRgbdFeatures(model, vis_layer, feature_type='rgb')
    ext_model = AlexnetMapFeatures(model, vis_layer)
    n_kernels = get_layer_width(ext_model)

    # AM on individual kernels
    for kernel_idx in range(n_kernels):
        paths.create_kernel_paths(vis_layer, str(kernel_idx))

        # Load Jacquard Dataset images
        print('Visualizing saliency maps')
        for img, map, img_id in data:
            # Create img folder
            #paths.create_img_paths(img_id)
            # Image preprocessing (None for now)
            prep_img = img
            
            GBP = GuidedBackprop(ext_model)
            # Get gradients
            guided_grads = GBP.generate_gradients(prep_img, kernel_idx)
            guided_grads = guided_grads[:3]
            # Convert to grayscale
            grayscale_guided_grads = convert_to_grayscale(guided_grads)
            # Save grayscale gradients
            save_gradient_images(grayscale_guided_grads, paths.save_subdir,
                                    '%s_%s_%s' % (img_id, vis_layer, kernel_idx) + '_Guided_BP_gray' )
            save_img_rgb, save_img_d = tensor2img(img)
            if '%s_image.png' % img_id not in os.listdir(os.path.join(paths.vis_path, paths.main_path)):
                cv2.imwrite(os.path.join(paths.vis_path, paths.main_path, '%s_image.png' % img_id), save_img_rgb)
    
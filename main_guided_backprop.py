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
import numpy as np
import torch

from parameters import Params
from paths import Path
from PIL import Image

from inference.models.alexnet import AlexnetMap_v3
from data_loader_v2 import DataLoader
from models import AlexnetMapRgbdFeatures, AlexnetMapFeatures
from guided_backprop import GuidedBackprop

from utils import tensor2img, get_layer_width
from misc_functions import (convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency,
                            save_image,
                            format_np_output)

# Params class containing parameters for AM visualization.
params = Params()
# Path class for managing required directories
paths = Path()
paths.create_guided_am_path()

# Select data for visualization
data = []
# DataLoader
dataLoader = DataLoader('../GrTrainer-paperspace/data/top_5_compressed/train', 1, 0.0, return_mask=True, device="cpu", seed=42)
for i, datum in enumerate(dataLoader.load_cls()):
    if i >= params.N_IMG:
        break
    data.append(datum)

# Initialized pretrained alexnetMap model
model = AlexnetMap_v3().to(params.DEVICE)
model.load_state_dict(torch.load('trained-models/%s/%s_epoch150.pth' % (params.MODEL_NAME, params.MODEL_NAME)))
model.eval()

for vis_layer in params.vis_layers:
    print('Visualizing for %s layer' % vis_layer)
    # Create sub-directory for chosen layer
    paths.create_layer_paths(vis_layer)
    
    # Create submodel with output = selected kernel
    #ext_model = AlexnetMapRgbdFeatures(model, vis_layer, feature_type='rgb')
    ext_model = AlexnetMapFeatures(model, vis_layer)
    n_kernels = get_layer_width(ext_model)

    # Load Jacquard Dataset images
    print('Visualizing saliency maps')
    for id, (img_rgbd, cls_map, img_cls_idx, img_mask) in enumerate(data):
        mean_guided_grads = np.zeros((1, 224, 224))
        # AM on individual kernels
        #for kernel_idx in range(n_kernels):
        # grasp top_k
        #[57, 5, 62, 53, 40, 50, 55, 56, 47, 44] - rgb_features.0
        #[54, 14, 24, 43, 61] - features.10
        #[53, 35, 49,  2, 63] - features-7
        for i, kernel_idx in enumerate([53, 35, 49,  2, 63]):
        # cls top_k
        #[6, 16,  9, 20, 38, 41, 47, 40, 37,  8] - rgb_features.0
        #[12, 31, 11, 22, 33] - features.10
        #[2, 51,  4, 18, 52] - features.7
        #for i, kernel_idx in enumerate([2, 51,  4, 18, 52]):
            #paths.create_kernel_paths(vis_layer, 'rank_%s_kernel_%s' % (10 - i, kernel_idx))

            # Create img folder
            #paths.create_img_paths(img_id)
            # Image preprocessing (None for now)
            prep_img = img_rgbd
            
            GBP = GuidedBackprop(ext_model)
            # Get gradients
            guided_grads = GBP.generate_gradients(prep_img, kernel_idx)
            guided_grads = guided_grads[:3]
            # Convert to grayscale
            grayscale_guided_grads = convert_to_grayscale(guided_grads)
            
            # Mask saliency
            grayscale_guided_grads = grayscale_guided_grads * np.array(img_mask)

            # Find value of gradient of rank 20th
            flattened_gradient = np.reshape(grayscale_guided_grads, (grayscale_guided_grads.shape[0], -1))
            top_k = int(len(flattened_gradient[0]) * 0.01)
            top_k_grad = np.sort(flattened_gradient, 1)[0][-top_k]
            # Subset gradient to only top 20
            grayscale_guided_grads = np.where(grayscale_guided_grads >= top_k_grad, grayscale_guided_grads, 0)

            mean_guided_grads = mean_guided_grads + grayscale_guided_grads

        grayscale_guided_grads = mean_guided_grads / 5
        save_img_rgb, save_img_d = tensor2img(img_rgbd)

        # Normalize and resize gradient map
        gradient = grayscale_guided_grads - grayscale_guided_grads.min()
        gradient /= gradient.max()
        gradient = gradient.transpose(1, 2, 0)
        gradient = cv2.resize(gradient, (save_img_rgb.shape[0], save_img_rgb.shape[1]), cv2.INTER_LINEAR)
        #gradient = gradient * np.array(img_mask)
        gradient = np.expand_dims(gradient, 2)

        # Save grayscale gradients
        save_gradient_images(gradient.transpose(2, 0, 1), paths.save_subdir,
                                'image_%s_layer_%s' % (id, vis_layer) + '_Guided_BP_gray' )

        # Define highlighting color
        sub_color = np.full((gradient.shape[0], gradient.shape[1], 1), 0.2)
        main_color = np.full((gradient.shape[0], gradient.shape[1], 1), 0.95)
        color1 = np.concatenate((main_color, sub_color, sub_color), 2)

        # Highlight image based on gradient
        colored_img = save_img_rgb * 0.5 + gradient * color1
        colored_img = np.clip(colored_img, 0, 255)
        
        # Save image
        path_to_file = os.path.join(paths.save_subdir, 'image_%s_layer_%s' % (id, vis_layer) + '_Guided_BP_img.png')
        #path_to_file = os.path.join(paths.save_subdir, 'image_%s_layer_%s_kernel_%s_rank_%s' % (id, vis_layer, kernel_idx, i) + '_Guided_BP_img.png')

        im = Image.fromarray(np.ascontiguousarray(colored_img, dtype=np.uint8))
        im = im.resize((256, 256), Image.ANTIALIAS)
        im.save(path_to_file)

        #if '%s_image.png' % img_id not in os.listdir(os.path.join(paths.vis_path, paths.main_path)):
        #    cv2.imwrite(os.path.join(paths.vis_path, paths.main_path, '%s_image.png' % img_id), save_img_rgb)

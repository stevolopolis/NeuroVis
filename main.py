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

from parameters import GrParams
from paths import GrPath
from models import ExtractModel

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
        start_img_path = '%s/%s_%s_start.png' % (paths.save_subdir, vis_layer, kernel_idx)
        end_img_path = '%s/%s_%s_end.png' % (paths.save_subdir, vis_layer, kernel_idx)
        target_path = '%s/%s_%s_target.png' % (paths.save_subdir, vis_layer, kernel_idx)
        output_path = '%s/%s_%s_output.png' % (paths.save_subdir, vis_layer, kernel_idx)

        # Load trained gr-convnet model
        model = torch.load(params.MODEL_PATH, map_location=params.DEVICE)
        model.eval()

        # Create submodel with output = selected kernel
        ext_model = ExtractModel(model, vis_layer, net_type=params.net, device=params.DEVICE)


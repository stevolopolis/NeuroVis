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

from parameters import GrParams
from paths import GrPath

# Params class containing parameters for AM visualization.
params = GrParams()
# Path class for managing required directories
# gr-convnet / resnet18 / vgg16
paths = GrPath('gr-convnet')

for vis_layer in params.vis_layers:
    print('Visualizing for %s layer' % vis_layer)
    paths.create_layer_paths(vis_layer)
    for kernel_idx in range(params.N_KERNELS):
        start_img_path = '%s/%s_%s_start.png' % (paths.save_subdir, vis_layer, kernel_idx)
        end_img_path = '%s/%s_%s_end.png' % (paths.save_subdir, vis_layer, kernel_idx)
        target_path = '%s/%s_%s_target.png' % (paths.save_subdir, vis_layer, kernel_idx)
        output_path = '%s/%s_%s_output.png' % (paths.save_subdir, vis_layer, kernel_idx)



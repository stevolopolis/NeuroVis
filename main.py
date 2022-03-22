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

# Params class containing parameters for AM visualization.
params = GrParams()

for vis_layer in params.vis_layers:
    print('Visualizing for %s layer' % vis_layer)

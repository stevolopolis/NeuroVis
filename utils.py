"""
This file contains some miscellaneous functions for
processing images, directories, or image generation, etc.
"""

import os
import shutil
import cv2
import numpy as np

from parameters import GrParams

params = GrParams()


# ===================================
# Tensor to image processing
# ===================================
def tensor2array(tensor):
    """Returns numpy arrays for from pytorch tensors."""
    
    img = tensor.detach().cpu().numpy()
    # if tensor is a 2-dimensional map (e.g. feature maps)
    # we repeat the map for 3 times for RGB channels
    if img.shape[1] == 1:
        img = img.repeat(1, 3, 1, 1)

    return img[0]


def array2img(img):
    """Returns numpy arrays compatible for cv2 visualization."""
    # normalization for vgg/resnet
    if params.net in ['vgg16', 'resnet18']:
        img = imagenet_norm(img)
        
    img = pixel_range_norm(img)
    # Swap RGB channel axis for compatibility with cv2
    img = np.moveaxis(img, 0, 2)

    # resize for small images
    img = cv.resize(img, (224, 224), interpolation=cv.INTER_LINEAR)

    return img.astype(int)


def slice_img(img):
    """Return rgb-image and depth-image given a
    4-dimensional image array."""

    rgb_img = img[:, :, :3]
    d_img = img[:, :, 3]

    return rgb_img, d_img


def imagenet_norm(img):
    """Return normalized image based on mean and
    std of imagenet dataset.
    
    Suitable for torchvision pretrained models.
    """

    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    for c in range(3):
        img[c] /= reverse_std[c]
        img[c] -= reverse_mean[c]

    return img


def pixel_range_norm(img):
    """Return normalized image with pixels ranging from 
    0 to 255.
    """
    
    # If the entire image is the same color,
    # we normalize the image to be a single-grey image
    if np.amin(img) == np.amax(img):
        img = img / (2 * np.amax(img))
    else:
        # Set pixel range to be >= 0.
        img = img - np.amin(img)
        # Set pixel range to be between 0 and 1.
        img = img / np.amax(img)

    # Scale pixel range to be bewteen 0 and 255.
    img = img * 255
    
    return img


# ===================================
# Directory management
# ===================================
def am_img_mat(imgs):
    """Return a 2x3 matrix of images."""
    pass


def clean_dir(dir):
    for name in os.listdir(dir):
        path = os.path.join(dir, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def check_dir(path):
    if os.path.exists(path):
        clean_dir(path)
    else:
        os.mkdir(path)


def get_angle_experimentation_pairs():
    return [(1500, 'rect', 'zero', 'sin', 0.1, 20),     # 20 degree anti-clockwise -- sine output -- zero initiatlization
            (1500, 'rect', 'zero', 'sin', 0.1, -20),    # 20 degree clockwise -- sine output -- zero initiatlization
            (1500, 'rect', 'zero', 'cos', 0.1, 20),     # 20 degree anti-clockwise -- cosine output -- zero initiatlization
            (1500, 'rect', 'zero', 'cos', 0.1, -20),    # 20 degree clockwise -- cosine output -- zero initiatlization
            (1500, 'rect', 'zero', 'sin', 0.1, 45),     # 45 degree anti-clockwise -- sine output -- zero initiatlization
            (1500, 'rect', 'zero', 'sin', 0.1, -45),    # 45 degree clockwise -- sine output -- zero initiatlization
            (1500, 'rect', 'zero', 'cos', 0.1, 90),     # 90 degree -- cosine output -- zero initialization
            (1500, 'rect', 'zero', 'cos', 0.1, 65),     # 65 degree anti-clockwise -- cosine output -- zero initiatlization
            (1500, 'rect', 'zero', 'cos', 0.1, -65),    # 65 degree clockwise -- cosine output -- zero initiatlization
            (1500, 'rect', 'zero', 'sin', 0.1, 65),     # 65 degree anti-clockwise -- sine output -- zero initiatlization
            (1500, 'rect', 'zero', 'sin', 0.1, -65)]    # 65 degree clockwise -- sine output -- zero initiatlization


def get_n_kernels_from_layer(layer_name):
    if layer_name == 'conv1':
        return 32
    elif layer_name == 'conv2':
        return 64
    elif layer_name in ['conv3', 'res1', 'res2', 'res3', 'res4', 'res5']:
        return 128


def get_variations_per_angle(rank, allow_neg=True):
    n_vars = len(rank)
    if allow_neg:
        n_vars_per_angle = n_vars // 18
    else:
        n_vars_per_angle = n_vars // 9
    return n_vars_per_angle
        
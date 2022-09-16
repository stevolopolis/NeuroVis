"""
This file contains some miscellaneous functions for
processing images, directories, or image generation, etc.
"""

import os
import torch
import shutil
import cv2
import math
import numpy as np

from PIL import Image

from parameters import GrParams

params = GrParams()


# ===================================
# Image augmentation
# ===================================
class AddGaussianNoise(torch.nn.Module):
    """Gaussian noise augmentation fn used in DataLoader class."""
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


# ======================================
# Model properties
# =========================================
def get_layer_width(model):
    dummy_x = torch.randn((1, params.IMG_SIZE[0], params.IMG_SIZE[1], params.IMG_SIZE[2]))
    out = model(dummy_x)
    layer_width = out.shape[1]  # (batch_size, n_kernels, img_dim[0], img_dim[1])

    return layer_width


# ===================================
# Tensor to image processing
# ===================================
def tensor2array(tensor):
    """Returns numpy arrays for from pytorch tensors."""
    
    # if tensor is a 2-dimensional map (e.g. feature maps)
    # we unsqueeze the map and repeat along RGB channel for
    # 3 times.
    if len(tensor.shape) == 2:
        tensor = torch.unsqueeze(tensor, 0)
        tensor = torch.unsqueeze(tensor, 0)
        tensor = tensor.repeat(1, 3, 1, 1)
    
    img = tensor.detach().cpu().numpy()

    return img[0]


def array2img(img, interpolation=cv2.INTER_AREA, size=params.vis_img_size):
    """Returns numpy arrays compatible for cv2 visualization."""
    # normalization for vgg/resnet
    if params.net in ['vgg16', 'resnet18']:
        img = imagenet_norm(img)
        
    img = pixel_range_norm(img)
    # Swap RGB channel axis for compatibility with cv2
    img = np.moveaxis(img, 0, 2)

    # resize for small images
    img = cv2.resize(img, size, interpolation=interpolation)

    return img.astype(int)


def slice_img(img):
    """Return rgb-image and depth-image given a
    4-dimensional image array."""

    rgb_img = img[:, :, :3]
    d_img = img[:, :, 3]

    return rgb_img, d_img


def expand_2d_img(img):
    """Return cv2.imwrite-ready 3D array given a 2D array."""
    img = np.tile(img, (3, 1, 1))
    img = np.moveaxis(img, 0, -1)

    return img


def tensor2img(img, interpolation=cv2.INTER_AREA):
    """Returns rgb-image and depth-image given tensor"""
    img = tensor2array(img)
    img = array2img(img, interpolation)
    if img.shape[2] == 4:
        rgb, d = slice_img(img)
        d = expand_2d_img(d)
        return rgb, d
    else:
        return img, img


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
    img = np.clip(img, 0, 255)
    
    return img


# ===================================
# Image processing
# ===================================
def am_img_mat(pixel_set, full_set):
    """Return a 2x3 (2x4 including empty col) matrix of images.

    Each element in imgs has the shape: (h, w, 3)
    
    Parameters:
        - set[0]: <start_img_rgb>
        - set[1]: <start_img_d>
        - set[2]: <backprop_img_rgb>
        - set[3]: <backprop_img_d>
        - set[4]: <fmap_img>
        - set[5]: <target_img>
    """
    col_h = params.vis_img_size[0] * 2
    col_w = params.vis_img_size[1]

    start_col = np.concatenate((pixel_set[0], pixel_set[1], full_set[0], full_set[1]), axis=0)
    backprop_col = np.concatenate((pixel_set[2], pixel_set[3], full_set[2], full_set[3]), axis=0)
    fmap_col = np.concatenate((pixel_set[4], pixel_set[5], full_set[4], full_set[5]), axis=0)

    img_mat = np.concatenate((start_col, backprop_col, fmap_col), axis=1)

    return img_mat


def fmap_img_mat(imgs):
    """Returns a matrix of feature-map images in the form of np.array."""
    n_imgs = len(imgs)
    h = math.log(n_imgs, 2) // 2
    w = n_imgs // h

    assert h * w == n_imgs

    col_set = []
    for i in range(h):
        col = np.concatenate(imgs[i * w : (i+1) * w], axis=0)
        col_set.append(col)

    img_mat = np.concatenate(col_set, axis=1)
    
    return img_mat


def get_real_img_from_path(path: str):
    """Returns image tensor and image id from RGB image path.
    
    Format of <path>:
        - '..\\<left/right>\\<img_id>_<img_type>.png'
        - <img_type> could be 'mask', 'rgb', 'depth', etc.
    """
    # Handling image file name
    subdir = path.split('\\')[-2]  # 'left' or 'right'
    file_name = path.split('\\')[-1]
    id = file_name[:-7]

    # Loading rgb and depth image
    img_rgb = np.array(Image.open(path))
    img_d_name = id + 'mask.png'
    img_d_path = os.path.join(params.DATA_PATH, subdir, img_d_name)
    img_d = np.array(Image.open(img_d_path))

    # Combine rgb and depth image
    img_d = np.expand_dims(img_d, 2)
    img = np.concatenate((img_rgb, img_d), axis=2)
    # Move color channel to match model requirement
    img = np.moveaxis(img, -1, 0)
    img = np.expand_dims(img, 0)
    img = torch.tensor(img, dtype=torch.float32).to(params.DEVICE)

    return img, id


# ===================================
# Directory management
# ===================================
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

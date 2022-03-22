import os
import shutil
import cv2 as cv
import numpy as np


def create_rgbd_image(img, normalize=False):
    img = img[0]
    
    # for vgg/resnet
    """reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    for c in range(3):
        img[c] /= reverse_std[c]
        img[c] -= reverse_mean[c]"""
        
    scale_factor = 255
    if normalize:
        if np.amin(img) != np.amax(img):
            img = img - np.amin(img) # set minimum of img to 0
        if np.amax(img) != 0:
            scale_factor = 255/np.amax(img)
    
    img = img * scale_factor
    img = np.moveaxis(img, 0, 2)

    # resize for small images
    img = cv.resize(img, (224, 224), interpolation=cv.INTER_LINEAR)

    img = img.astype(int)
    img = np.clip(img, 0, 255)
    if img.shape[2] == 4:
        rgb_img = img[:, :, :3]
        d_img = img[:, :, 3]
        return rgb_img, d_img
    elif img.shape[2] == 3:
        rgb_img = img[:, :, :3]
        return rgb_img


def output_set_prep(start_img, end_img, target, output):
    start_img = start_img.detach().cpu().numpy()
    end_img = end_img.detach().cpu().numpy()
    target = target.repeat(1, 3, 1, 1)
    output = output.repeat(1, 3, 1, 1)
    target = target.detach().cpu().numpy()
    output = output.detach().cpu().numpy()

    return start_img, end_img, target, output


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


def get_variations_per_angle(rank, allow_neg=True):
    n_vars = len(rank)
    if allow_neg:
        n_vars_per_angle = n_vars // 18
    else:
        n_vars_per_angle = n_vars // 9
    return n_vars_per_angle


def display_analysis_details():
    print('Something')
        
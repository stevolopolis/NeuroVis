"""
This file contains the DataLoader class which is responsible for
loading both CLS data and Grasping data.

DataLoader also includes all the necessary function for data augmentation
such as a color and noise augmentation pipeline for CLS and
rotation for Grasping.

"""
import glob
import torch
import os
import random
import math
import torch.nn as nn
import numpy as np

from PIL import Image
from torchvision import transforms
from parameters import Params
from utils import AddGaussianNoise, tensor_concat

params = Params()

class MyRotationTransform:
    """Rotate by one of the given angles."""
    def __init__(self, angle):
        self.angles = angle

    def __call__(self, x):
        return transforms.functional.rotate(x, self.angle)


class DataLoader:
    """
    DataLoader class. Loads both CLS data and Grasping data.

    CLS data:
        - self.load_batch() and self.load()
    Grasp data:
        - self.load_grasp_batch() and self.load_grasp()
    Image processing:
        - self.process()
    CLS labels:
        - self.scan_img_id() and self.get_cls_id()
    Grasp labels:
        - self.load_grasp_label() and self.get_grasp_label()
    """
    def __init__(self, path, batch_size, train_val_split=0.2, include_depth=True, return_mask=False, verbose=True, seed=None, device=params.DEVICE):
        self.path = path
        self.batch_size = batch_size
        self.train_val_split = train_val_split
        self.return_mask = return_mask
        self.include_depth = include_depth
        self.device = device

        # Get list of class names
        self.img_cls_list = self.get_cls_id()
        # Get dictionary of image-id to classes
        self.img_id_map = self.scan_img_id(verbose=verbose)
        self.n_data = len(self.img_id_map.keys())
        self.img_id_list = list(self.img_id_map.keys())
        # Shuffle ids for training
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.img_id_list)

        # Custom data augmentations
        # Add gaussian noise with 25% probability
        random_transforms = transforms.RandomApply(nn.ModuleList([AddGaussianNoise(0, .02, device=self.device)]), p=0.25)
        # Color data augmentations
        self.transformation_rgb = transforms.Compose([
            #transforms.ColorJitter(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
            #random_transforms
            #transforms.Grayscale(num_output_channels=3)
        ])
        

    def load_batch(self):
        """Yields a batch of CLS training data -- (img, label)."""
        for i, (img, cls_map, label) in enumerate(self.load_cls(include_depth=self.include_depth)):
            if i % self.batch_size == 0:
                img_batch = img
                map_batch = cls_map
                label_batch = label
            elif (i+1) % self.batch_size == 0:
                img_batch = torch.cat((img_batch, img), dim=0)
                map_batch = torch.cat((map_batch, cls_map), dim=0)
                label_batch = torch.cat((label_batch, label), dim=0)
                yield (img_batch, map_batch, label_batch)
            else:
                img_batch = torch.cat((img_batch, img), dim=0)
                map_batch = torch.cat((map_batch, cls_map), dim=0)
                label_batch = torch.cat((label_batch, label), dim=0)
            
        # This line catches the final few instances (less than batch_size)
        if (i + 1) % self.batch_size != 0:
            yield (img_batch, map_batch, label_batch)
    
    def load_cls(self, include_depth=True):
        """Yields a single instance of CLS training data -- (img, label)."""
        for img_id_with_var in self.img_id_list:
            img_angle = int(img_id_with_var.split('_')[-1])
            img_id = img_id_with_var.split('_')[-2]
            img_var = img_id_with_var.split('_')[0]
            img_name = img_var + '_' + img_id
            img_cls = self.img_id_map[img_id_with_var]
            img_cls_idx = self.img_cls_list.index(img_cls)
            img_cls_idx = torch.tensor([img_cls_idx]).to(self.device)

            label = torch.ones(6, dtype=torch.float32) * -1
            label[img_cls_idx] = 1.0
            label[5] = 1.0

            img_path = os.path.join(self.path, img_cls, img_id)

            # Open RGB npy file
            img_rgb = np.load(open(os.path.join(img_path, img_name + '_RGB.npy'), 'rb'))
            img_rgb = torch.tensor(img_rgb, dtype=torch.float32).to(self.device)
            # Open Depth npy file
            img_d = np.load(open(os.path.join(img_path, img_name + '_perfect_depth.npy'), 'rb'))
            img_d = torch.tensor(img_d, dtype=torch.float32).to(self.device)
            # Open Mask npy file
            img_mask = np.load(open(os.path.join(img_path, img_name + '_mask.npy'), 'rb'))
            img_mask = torch.tensor(img_mask, dtype=torch.float32).to(self.device)

            cls_map = mask_to_cls_map(img_mask, label)
            cls_map = torch.unsqueeze(cls_map, 0).to(self.device)

            # Normalize and combine rgb with depth channel
            img_rgbd = self.process(img_rgb, img_d, include_depth=include_depth)

            if img_angle != 0:
                img_rgbd = transforms.functional.rotate(img_rgbd, img_angle)
                cls_map = transforms.functional.rotate(cls_map, img_angle)
                img_mask = transforms.functional.rotate(torch.unsqueeze(img_mask, 0), img_angle)

            if not self.return_mask:
                yield (img_rgbd, cls_map, img_cls_idx)
            else:
                yield (img_rgbd, cls_map, img_cls_idx, torch.squeeze(img_mask, 0))

    def load_grasp_batch(self):
        """Yields a batch of Grasp training data -- (img, grasp-label, grasp-candidates)."""
        for i, (img, grasp_map, grasp_list) in enumerate(self.load_grasp()):
            if i % self.batch_size == 0:
                img_batch = img
                grasp_map_batch = grasp_map
                grasp_list_batch = torch.unsqueeze(grasp_list, dim=0)
            elif (i+1) % self.batch_size == 0:
                img_batch = torch.cat((img_batch, img), dim=0)
                grasp_map_batch = torch.cat((grasp_map_batch, grasp_map), dim=0)
                grasp_list_batch = tensor_concat(grasp_list_batch, torch.unsqueeze(grasp_list, dim=0))
                yield (img_batch, grasp_map_batch, grasp_list_batch)
            else:
                img_batch = torch.cat((img_batch, img), dim=0)
                grasp_map_batch = torch.cat((grasp_map_batch, grasp_map), dim=0)
                grasp_list_batch = tensor_concat(grasp_list_batch, torch.unsqueeze(grasp_list, dim=0))
            
        # This line of code catches the final few instances (less that batch_size)
        if (i + 1) % self.batch_size != 0:
            yield (img_batch, grasp_map_batch, grasp_list_batch)

    def load_grasp(self, include_depth=True):
        """Yields a single instance of Grasp training data -- (img, grasp-map)."""
        for img_id_with_var in self.img_id_list:
            img_angle = int(img_id_with_var.split('_')[-1])
            img_id = img_id_with_var.split('_')[-2]
            img_var = img_id_with_var.split('_')[0]
            img_name = img_var + '_' + img_id
            img_cls = self.img_id_map[img_id_with_var]

            img_path = os.path.join(self.path, img_cls, img_id)

            # Open RGB image with PIL
            img_rgb = np.load(open(os.path.join(img_path, img_name + '_RGB.npy'), 'rb'))
            img_rgb = torch.tensor(img_rgb, dtype=torch.float32).to(self.device)
            # Open Depth image with PIL
            img_d = np.load(open(os.path.join(img_path, img_name + '_perfect_depth.npy'), 'rb'))
            img_d = torch.tensor(img_d, dtype=torch.float32).to(self.device)

            # Get Grasp map
            grasp_map = np.load(open(os.path.join(img_path, img_name + '_' + str(img_angle) + '_map_grasps.npy'), 'rb'))
            grasp_map = torch.tensor(grasp_map).to(self.device)
            grasp_map = self.normalize_grasp_map(grasp_map)
            # Get Grasp list
            grasp_list = np.load(open(os.path.join(img_path, img_name + '_' + str(img_angle) + '_txt_grasps.npy'), 'rb'))
            grasp_list = torch.tensor(grasp_list).to(self.device)
            grasp_list = self.normalize_grasp_arr(grasp_list)

            # Normalize and combine rgb with depth channel
            img_rgbd = self.process(img_rgb, img_d, include_depth=include_depth)

            # Augmentation on image -- random rotations (can only do 1/2 pi rotations for label accuracy)
            if img_angle != 0:
                img_rgbd = transforms.functional.rotate(img_rgbd, img_angle)
            
            yield (img_rgbd, grasp_map, grasp_list)
        
    def process(self, rgb, d, include_depth=True):
        """
        Returns rgbd image with correct format for inputing to model:
            - Imagenet normalization
            - Concat depth channel to image
        """
        rgb = rgb / 255.0
        rgb = torch.moveaxis(rgb, -1, 0)
        rgb = self.transformation_rgb(rgb)
        if include_depth:
            # Input channels -- (red, green, blue, depth)
            d = torch.unsqueeze(d, 2)
            d = d - torch.mean(d)
            d = torch.clip(d, -1, 1)
            d = torch.moveaxis(d, -1, 0)
            img = torch.cat((rgb, d), axis=0)
        else:
            # rgb
            img = rgb
            # depth
            #d = torch.unsqueeze(d, 2)
            #d = d - torch.mean(d)
            #d = torch.clip(d, -1, 1)
            #d = torch.moveaxis(d, -1, 0)
            #img = torch.cat((d, d, d), axis=0)

        img = torch.unsqueeze(img, 0)
        img = img.to(self.device)

        return img

    def normalize_grasp_map(self, grasp_map):
        """Returns normalize grasping labels."""
        # Normalize x-coord
        grasp_map[:, :, 0] /= 224
        # Normalize y-coord
        grasp_map[:, :, 1] /= 224
        # Normalize width
        grasp_map[:, :, 3] /= 244
        # Normalize height (range: [-1, 1])
        grasp_map[:, :, 4] = (grasp_map[:, :, 4] - 0.5) * 2
        # Reshape to match input dim
        grasp_map = torch.unsqueeze(grasp_map, 0)
        grasp_map = torch.moveaxis(grasp_map, -1, 1)

        return grasp_map
        
    def normalize_grasp_arr(self, label):
        """Returns normalize grasping labels."""
        # Normalize x-coord
        label[:, 0] /= 224
        # Normalize y-coord
        label[:, 1] /= 224
        # Normalize width
        label[:, 3] /= 224

        return label

    def load_grasp_label(self, file_path):
        """Returns a list of grasp labels from <file_path>."""
        grasp_list = []
        with open(file_path, 'r') as f:
            file = f.readlines()
            # dat format in each line: 'x;y;theta;w;h'
            for grasp in file:
                # remove '\n' from string
                grasp = grasp[:-1]
                label = grasp.split(';')
                label = self.noramlize_grasp_old(label)
                grasp_list.append(label)

        return grasp_list

    def scan_img_id(self, verbose=True):
        """
        Returns a dictionary mapping the image ids from the 'data' 
        folder to their corresponding classes.

        '/' (linux) may have to be changed to '\\' (windows).
        """
        img_id_dict = {}
        for img_path in glob.iglob('%s/*/*/*' % self.path):
            if not img_path.endswith('map_grasps.npy'):
                continue
            
            img_cls = img_path.split('\\')[-3]
            # E.g. '<img_idx>_<img_id>_<angle>_<img_type>.png'
            img_name = img_path.split('\\')[-1]
            img_var = img_name.split('_')[0]
            img_id = img_name.split('_')[1]
            img_angle = img_name.split('_')[-3]
            img_id_with_var = img_var + '_' + img_id + '_' + img_angle
            img_id_dict[img_id_with_var] = img_cls

        n_data = len(img_id_dict.keys())
        n_train, n_val = self.get_train_val(n_data)
        #print("debug1:",img_id_dict)
        if verbose:
            print('Dataset size: %s' % n_data)
            print('Training steps: %s -- Val steps: %s' % (n_train, n_val))
        return img_id_dict

    def get_cls_id(self):
        """Returns a list of class names in fixed order (according to the txt file)."""
        cls_list = []
        with open(os.path.join(params.DATA_PATH, params.LABEL_FILE), 'r') as f:
            file = f.readlines()
            for cls in file:
                # remove '\n' from string
                cls = cls[:-1]
                cls_list.append(cls)
        return cls_list

    def get_train_val(self, n_data=None):
        """Returns the number of training/validation steps."""
        if n_data is not None:
            n_steps = math.ceil(n_data / self.batch_size)
        else:
            n_steps = math.ceil(self.n_data / self.batch_size)
        n_val = round(n_steps * self.train_val_split)
        n_train = n_steps - n_val
        return n_train, n_val


# ----------------------------------------------------------------
# Geometric augmentations for Grasp data
# ----------------------------------------------------------------
def crop_jitter_resize(img, ratio, jitter_x, jitter_y):
    """
    Returns an augmented image after crop-jitter-resizing.
    Not used in current training pipeline.
    """
    # img.shape = (1, 3, img_h, img_w)
    img_h = img.shape[2]
    img_w = img.shape[3]
    new_img_h = int(img_h * ratio)
    new_img_w = int(img_w * ratio)

    jitter_coord_y = int(jitter_y * params.OUTPUT_SIZE)
    jitter_coord_x = int(jitter_x * params.OUTPUT_SIZE)

    crop_y = (img_h - new_img_h) // 2
    crop_x = (img_w - new_img_w) // 2
    
    crop_img = img[:, :, crop_y + jitter_coord_y : crop_y + new_img_h + jitter_coord_y,\
                   crop_x + jitter_coord_x : crop_x + new_img_w + jitter_coord_x]

    return transforms.functional.resize(crop_img, (img_h, img_w))


# --------------------------------
def mask_to_cls_map(img_mask, label):
    """Return cls map using mask image."""
    img_mask = img_mask.cpu()
    img_mask = torch.unsqueeze(img_mask, 2)
    img_mask = torch.cat((img_mask, img_mask, img_mask, img_mask, img_mask, img_mask), 2)
    background_val = img_mask[0][0]
    background_mask = torch.ones((224, 224, 6)) * -1
    background_mask[:, :, 5] = 0.0
    cls_map = torch.where(img_mask == background_val, background_mask, label.cpu())
    cls_map = torch.moveaxis(cls_map, -1, 0)

    return cls_map
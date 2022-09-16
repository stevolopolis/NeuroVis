"""
This file contains the DataLoader class which is responsible for
loading both CLS data and Grasping data.

DataLoader also includes all the necessary function for data augmentation
such as a color and noise augmentation pipeline for CLS and a
rotation+translation pipeline for Grasping.

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
from parameters import GrParams
from utils import AddGaussianNoise

params = GrParams()

class MyRotationTransform:
    """Rotate by one of the given angles."""
    def __init__(self, angle):
        self.angles = angle

    def __call__(self, x):
        return transforms.functional.rotate(x, self.angle)


class DataLoaderV2:
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
    def __init__(self, path, batch_size, train_val_split=0.2):
        self.path = path
        self.batch_size = batch_size
        self.train_val_split = train_val_split

        # Get list of class names
        self.img_cls_list = self.get_cls_id()
        # Get dictionary of image-id to classes
        self.img_id_map = self.scan_img_id()
        self.n_data = len(self.img_id_map.keys())
        self.img_id_list = list(self.img_id_map.keys())
        # Shuffle ids for training
        random.shuffle(self.img_id_list)

        # Custom data augmentations
        # Add gaussian noise with 25% probability
        random_transforms = transforms.RandomApply(nn.ModuleList([AddGaussianNoise(0, .02)]), p=0.25)
        # Geometric data augmentations
        self.transformation = transforms.Compose([
            transforms.RandomResizedCrop(params.OUTPUT_SIZE, scale=(.75, .85), ratio=(1, 1)),
            transforms.RandomRotation(90)
        ])
        # Color data augmentations
        self.transformation_rgb = transforms.Compose([
            transforms.ColorJitter(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            random_transforms
            #transforms.Grayscale(num_output_channels=3)
        ])
        

    def load_batch(self):
        """Yields a batch of CLS training data -- (img, label)."""
        for i, (img, cls_map, label) in enumerate(self.load_cls()):
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
    
    def load_cls(self):
        """Yields a single instance of CLS training data -- (img, label)."""
        for img_id_with_var in self.img_id_list:
            img_angle = int(img_id_with_var.split('_')[-1])
            img_id = img_id_with_var.split('_')[-2]
            img_var = img_id_with_var.split('_')[0]
            img_name = img_var + '_' + img_id
            img_cls = self.img_id_map[img_id_with_var]
            img_cls_idx = self.img_cls_list.index(img_cls)
            img_cls_idx = torch.tensor([img_cls_idx]).to(params.DEVICE)

            label = torch.zeros(6, dtype=torch.float32)
            label[img_cls_idx] = 1.0
            label[5] = 1.0

            img_path = os.path.join(self.path, img_cls, img_id)

            # Open RGB image with PIL
            img_rgb = np.load(open(os.path.join(img_path, img_name + '_RGB.npy'), 'rb'))
            img_rgb = torch.tensor(img_rgb, dtype=torch.float32).to(params.DEVICE)
            # Open Depth image with PIL
            img_d = np.load(open(os.path.join(img_path, img_name + '_perfect_depth.npy'), 'rb'))
            img_d = torch.tensor(img_d, dtype=torch.float32).to(params.DEVICE)

            cls_map = depth_to_cls_map(img_d, label)
            cls_map = torch.unsqueeze(cls_map, 0).to(params.DEVICE)

            # Normalize and combine rgb with depth channel
            img_rgbd = self.process(img_rgb, img_d)
            """if img_angle != 0:
                # Augmentation on image -- random rotations (can only do 1/2 pi rotations for label accuracy)
                img_rgbd = transforms.functional.rotate(img_rgbd, img_angle)
                cls_map = transforms.functional.rotate(cls_map, img_angle)
                
                img_vis = torch.where(cls_map[0, :, :, 5] == 1.0, 255, 0)
                img_vis = torch.unsqueeze(img_vis, 2)
                img_vis = torch.cat((img_vis, img_vis, img_vis), 2)
                img_vis = img_vis.detach().cpu().numpy()
                img_vis = np.ascontiguousarray(img_vis, dtype=np.uint8)
                
                cv2.imshow('cls_map', img_vis)
                cv2.waitKey(0)"""

            yield (img_rgbd, cls_map, img_cls_idx)

    def load_grasp_batch(self):
        """Yields a batch of Grasp training data -- (img, grasp-label, grasp-candidates)."""
        for i, (img, grasp_map) in enumerate(self.load_grasp()):
            if i % self.batch_size == 0:
                img_batch = img
                grasp_map_batch = grasp_map
            elif (i+1) % self.batch_size == 0:
                img_batch = torch.cat((img_batch, img), dim=0)
                grasp_map_batch = torch.cat((grasp_map_batch, grasp_map), dim=0)
                yield (img_batch, grasp_map_batch)
            else:
                img_batch = torch.cat((img_batch, img), dim=0)
                grasp_map_batch = torch.cat((grasp_map_batch, grasp_map), dim=0)

        # This line of code catches the final few instances (less that batch_size)
        if (i + 1) % self.batch_size != 0:
            yield (img_batch, grasp_map_batch)

    def load_grasp(self):
        """Yields a single instance of Grasp training data -- (img, grasp-label, grasp-candidates)."""
        for img_id_with_var in self.img_id_list:
            img_angle = int(img_id_with_var.split('_')[-1])
            img_id = img_id_with_var.split('_')[-2]
            img_var = img_id_with_var.split('_')[0]
            img_name = img_var + '_' + img_id
            img_cls = self.img_id_map[img_id_with_var]

            img_path = os.path.join(self.path, img_cls, img_id)

            # Open RGB image with PIL
            img_rgb = np.load(open(os.path.join(img_path, img_name + '_RGB.npy'), 'rb'))
            img_rgb = torch.tensor(img_rgb, dtype=torch.float32).to(params.DEVICE)
            # Open Depth image with PIL
            img_d = np.load(open(os.path.join(img_path, img_name + '_perfect_depth.npy'), 'rb'))
            img_d = torch.tensor(img_d, dtype=torch.float32).to(params.DEVICE)

            # Get Grasp map
            grasp_map = np.load(open(os.path.join(img_path, img_name + '_' + str(img_angle) + '_grasps.npy'), 'rb'))
            grasp_map = torch.tensor(grasp_map).to(params.DEVICE)
            grasp_map = self.noramlize_grasp(grasp_map)
            
            # Normalize and combine rgb with depth channel
            img_rgbd = self.process(img_rgb, img_d)

            # Manual augmentaion random parameters
            ratio = random.uniform(0.75, 0.85)
            jitter_x = random.uniform(0.075, 0.075)
            jitter_y = random.uniform(-0.075, 0.075)
            
            # Augmentation on image -- random resized crops with jitters
            #img_rgbd = crop_jitter_resize(img_rgbd, ratio, jitter_x, jitter_y)
            if img_angle != 0:
                # Augmentation on image -- random rotations (can only do 1/2 pi rotations for label accuracy)
                img_rgbd = transforms.functional.rotate(img_rgbd, img_angle)
            # Augmentation on grasp map -- random resized crop with jitters
            #grasp_map = crop_jitter_resize(grasp_map, ratio, jitter_x, jitter_y)
            # Augmentation on grasp map -- random rotations (can only do 1/2 pi rotations for label accuracy)
            #grasp_map = transforms.functional.rotate(grasp_map, degree)
            
            yield (img_rgbd, grasp_map, img_name)
        
    def process(self, rgb, d):
        """
        Returns rgbd image with correct format for inputing to model:
            - Imagenet normalization
            - Concat depth channel to image
        """
        rgb = rgb / 255.0
        rgb = torch.moveaxis(rgb, -1, 0)
        rgb = self.transformation_rgb(rgb)
        if d is None:
            img = rgb
        elif params.N_CHANNELS == 3:
            # Input channels -- (gray, gray, depth)
            #rgb = transforms.Grayscale(num_output_channels=1)(rgb)
            #rgb = torch.cat((rgb, rgb), axis=0)
            # Input channels -- (red, green, depth)
            d = torch.unsqueeze(d, 2)
            d = d - torch.mean(d)
            d = torch.clip(d, -1, 1)
            d = torch.moveaxis(d, -1, 0)
            img = torch.cat((rgb[:2], d), axis=0)
        else:
            # Input channels -- (red, green, blue, depth)
            d = torch.unsqueeze(d, 2)
            d = d - torch.mean(d)
            d = torch.clip(d, -1, 1)
            d = torch.moveaxis(d, -1, 0)
            img = torch.cat((rgb, d), axis=0)

        img = torch.unsqueeze(img, 0)
        img = img.to(params.DEVICE)

        return img

    def noramlize_grasp(self, grasp_map):
        """Returns normalize grasping labels."""
        # Normalize x-coord
        grasp_map[:, :, 0] /= params.OUTPUT_SIZE
        # Normalize y-coord
        grasp_map[:, :, 1] /= params.OUTPUT_SIZE
        # Normalize width
        grasp_map[:, :, 3] /= params.OUTPUT_SIZE
        # Reshape to match input dim
        grasp_map = torch.unsqueeze(grasp_map, 0)
        grasp_map = torch.moveaxis(grasp_map, -1, 1)

        return grasp_map

    def scan_img_id(self):
        """
        Returns a dictionary mapping the image ids from the 'data' 
        folder to their corresponding classes.
        """
        img_id_dict = {}
        for img_path in glob.iglob('%s/*/*/*' % self.path):
            if not img_path.endswith('grasps.npy'):
                continue
            
            img_cls = img_path.split('\\')[-3]
            # E.g. '<img_idx>_<img_id>_<angle>_<img_type>.png'
            img_name = img_path.split('\\')[-1]
            img_var = img_name.split('_')[0]
            img_id = img_name.split('_')[1]
            img_angle = img_name.split('_')[-2]
            img_id_with_var = img_var + '_' + img_id + '_' + img_angle
            img_id_dict[img_id_with_var] = img_cls

        n_data = len(img_id_dict.keys())
        n_train, n_val = self.get_train_val(n_data)
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
    """Returns an augmented image after crop-jitter-resizing."""
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
def depth_to_cls_map(img_d, label):
    """Return cls map with depth image and given cls image."""
    for row_idx in range(len(img_d)):
        if row_idx >= 80:
            img_d[row_idx] += 0.000691 * 79 - 0.0000005 * ((row_idx/2)**2) + 0.000649 * (row_idx - 79)
        else:
            img_d[row_idx] += 0.00069 * row_idx - 0.0000005 * ((row_idx/2)**2)
        
    img_d = torch.where(torch.abs(img_d - 1.5740) <= 0.005, img_d[0][0], img_d)

    depth_map = img_d.cpu()
    depth_map = torch.unsqueeze(depth_map, 2)
    depth_map = torch.cat((depth_map, depth_map, depth_map, depth_map, depth_map, depth_map), 2)
    background_val = img_d[0][0].cpu()
    background_mask = torch.zeros((params.OUTPUT_SIZE, params.OUTPUT_SIZE, 6))
    cls_map = torch.where(depth_map == background_val, background_mask, label.cpu())
    cls_map = torch.moveaxis(cls_map, -1, 0)

    return cls_map
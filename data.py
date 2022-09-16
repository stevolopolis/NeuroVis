import glob
import torch
import os
import random
import numpy as np

from torchvision import transforms
from PIL import Image
from imageio import imread
from skimage.transform import resize
from parameters import GrParams
from roboticGrasp.dataset_processing import grasp

# Params class containing parameters for all visualization.
params = GrParams()

class ClsDataLoader:
    def __init__(self, randomize=True):
        random.seed(42)
        self.path = params.DATA_PATH

        # Get class names list
        self.img_cls_list = self.get_cls_id()
        # Get all ids in dataset
        self.img_id_map = self.scan_img_id()
        self.n_data = len(self.img_id_map.keys())
        self.img_id_list = list(self.img_id_map.keys())
        if randomize:
            # Shuffle ids for training
            random.shuffle(self.img_id_list)

        self.transformation = transforms.Compose([
            transforms.RandomResizedCrop(params.OUTPUT_SIZE, scale=(.75, .8), ratio=(1, 1))
        ])
        self.transformation_rgb = transforms.Compose([
            #transforms.Grayscale(num_output_channels=3)
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def load(self):
        for img_id_with_var in self.img_id_list:
            img_id = img_id_with_var.split('_')[-1]
            img_cls = self.img_id_map[img_id_with_var]
            img_cls_idx = self.img_cls_list.index(img_cls)
            img_cls_idx = torch.tensor([img_cls_idx])

            img_path = os.path.join(self.path, img_cls, img_id)

            # Open RGB image with PIL
            img_rgb = Image.open(os.path.join(img_path, img_id_with_var + '_RGB.png'))
            img_rgb = img_rgb.resize((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
            img_rgb = torch.tensor(np.array(img_rgb), dtype=torch.float32)
            # Open depth image with PIL
            img_d = Image.open(os.path.join(img_path, img_id_with_var + '_perfect_depth.tiff'))
            img_d = img_d.resize((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
            img_d = torch.tensor(np.array(img_d), dtype=torch.float32)

            if params.N_CHANNELS == 4:
                yield (self.process(img_rgb, img_d), img_id_with_var, img_cls_idx)
            elif params.N_CHANNELS == 3:
                yield (self.process_rgb(img_rgb), img_id_with_var, img_cls_idx)

    def process(self, rgb, d):
        """Returns rgbd image with correct format for inputing to model."""
        rgb = rgb / 255.0
        #rgb = rgb - torch.mean(rgb)
        rgb = torch.moveaxis(rgb, -1, 0)
        rgb = self.transformation_rgb(rgb)
        d = torch.unsqueeze(d, 2)
        d = d - torch.mean(d)
        d = torch.clip(d, -1, 1)
        d = torch.moveaxis(d, -1, 0)
        img = torch.cat((rgb, d), axis=0)
        img = torch.unsqueeze(img, 0)
        img = img.to(params.DEVICE)

        return self.transformation(img)

    def process_rgb(self, rgb):
        """Returns rgb image with correct format for inputing to model."""
        rgb = rgb / 255.0
        #rgb = rgb - torch.mean(rgb)
        rgb = torch.moveaxis(rgb, -1, 0)
        rgb = self.transformation_rgb(rgb)
        rgb = torch.unsqueeze(rgb, 0)
        rgb = rgb.to(params.DEVICE)
        return rgb

    def scan_img_id(self):
        """Returns all the ids of images present in the 'data' folder."""
        img_id_dict = {}
        for img_path in glob.iglob('%s/*/*/*' % self.path):
            if not img_path.endswith('RGB.png'):
                continue
            
            img_cls = img_path.split('\\')[-3]
            # E.g. '<img_idx>_<img_id>_<img_type>.png'
            img_name = img_path.split('\\')[-1]
            img_var = img_name.split('_')[0]
            img_id = img_name.split('_')[1]
            img_id_with_var = img_var + '_' + img_id
            img_id_dict[img_id_with_var] = img_cls

        n_data = len(img_id_dict.keys())
        print('Dataset size: %s' % n_data)
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


class GraspDataLoader:
    def __init__(self, randomize=True):
        random.seed(42)
        self.path = params.DATA_PATH

        # Get all ids in dataset
        self.img_id_map = self.scan_img_id()
        self.n_data = len(self.img_id_map.keys())
        self.img_id_list = list(self.img_id_map.keys())
        # Shuffle ids for training
        if randomize:
            random.shuffle(self.img_id_list)

        # Data augmentation
        self.transformation_rgb = transforms.Compose([
            #transforms.ColorJitter(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def load_grasp(self):
        for img_id_with_var in self.img_id_list:
            img_id = img_id_with_var.split('_')[-1]
            img_cls = self.img_id_map[img_id_with_var]

            img_path = os.path.join(self.path, img_cls, img_id)

            # Open RGB image with PIL
            img_rgb = Image.open(os.path.join(img_path, img_id_with_var + '_RGB.png'))
            img_rgb = img_rgb.resize((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
            img_rgb = torch.tensor(np.array(img_rgb), dtype=torch.float32)
            # Open depth image with PIL
            img_d = Image.open(os.path.join(img_path, img_id_with_var + '_perfect_depth.tiff'))
 
            img_d = img_d.resize((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
            img_d = torch.tensor(np.array(img_d), dtype=torch.float32)

            # Open grasp.txt file 
            grasp_file_path = os.path.join(img_path, img_id_with_var + '_grasps.txt')
            grasp_list = self.load_grasp_label(grasp_file_path)
            grasp_label = self.get_grasp_label(grasp_list, metric='random')
            grasp_label = np.array([grasp_label])
            
            # Normalize and combine rgb with depth channel
            img_rgbd = self.process(img_rgb, None)

            # Manual augmentaion random parameters
            degree = random.choice([0, 90, -90, 180])
            ratio = random.uniform(0.8, 0.85)
            jitter = 0 #random.uniform(-0.05, 0.05)
            
            img_rgbd = crop_jitter_resize_img(img_rgbd, ratio, jitter, jitter)
            #img_rgbd = transforms.functional.rotate(img_rgbd, degree)
            grasp_label = crop_jitter_resize_label(grasp_label, ratio, jitter, jitter)
            #grasp_label = rotate_grasp_label(grasp_label, degree)
            grasp_list = crop_jitter_resize_label(grasp_list, ratio, jitter, jitter)
            #grasp_list = rotate_grasp_label(grasp_list, degree)
            
            yield (img_rgbd,
                   torch.tensor(grasp_label, dtype=torch.float32).to(params.DEVICE),
                   torch.tensor(grasp_list).to(params.DEVICE),
                   img_id_with_var)    

    def process(self, rgb, d):
        """Returns rgbd image with correct format for inputing to model."""
        rgb = rgb / 255.0
        #rgb = rgb - torch.mean(rgb)
        rgb = torch.moveaxis(rgb, -1, 0)
        rgb = self.transformation_rgb(rgb)
        if d is None:
            img = rgb
        elif params.NUM_CHANNEL == 3:
            #rgb = transforms.Grayscale(num_output_channels=1)(rgb)
            #rgb = torch.cat((rgb, rgb), axis=0)
            d = torch.unsqueeze(d, 2)
            d = d - torch.mean(d)
            d = torch.clip(d, -1, 1)
            d = torch.moveaxis(d, -1, 0)
            img = torch.cat((rgb[:2], d), axis=0)
        else:
            d = torch.unsqueeze(d, 2)
            d = d - torch.mean(d)
            d = torch.clip(d, -1, 1)
            d = torch.moveaxis(d, -1, 0)
            img = torch.cat((rgb, d), axis=0)

        img = torch.unsqueeze(img, 0)
        img = img.to(params.DEVICE)

        return img

    def scan_img_id(self):
        """Returns all the ids of images present in the 'data' folder."""
        img_id_dict = {}
        for img_path in glob.iglob('%s/*/*/*' % self.path):
            if not img_path.endswith('RGB.png'):
                continue
            
            img_cls = img_path.split('\\')[-3]
            # E.g. '<img_idx>_<img_id>_<img_type>.png'
            img_name = img_path.split('\\')[-1]
            img_var = img_name.split('_')[0]
            img_id = img_name.split('_')[1]
            img_id_with_var = img_var + '_' + img_id
            img_id_dict[img_id_with_var] = img_cls

        n_data = len(img_id_dict.keys())
        print('Dataset size: %s' % n_data)
        return img_id_dict

    def load_grasp_batch(self):
        for i, (img, label, candidates) in enumerate(self.load_grasp()):
            #img = self.transformation(img)
            if i % self.batch_size == 0:
                img_batch = img
                label_batch = label
                candidate_batch = [candidates]
            elif (i+1) % self.batch_size == 0:
                img_batch = torch.cat((img_batch, img), dim=0)
                label_batch = torch.cat((label_batch, label), dim=0)
                candidate_batch.append(candidates)
                yield (img_batch, label_batch, candidate_batch)
            else:
                img_batch = torch.cat((img_batch, img), dim=0)
                label_batch = torch.cat((label_batch, label), dim=0)
                candidate_batch.append(candidates)

        if (i + 1) % self.batch_size != 0:
            yield (img_batch, label_batch, candidate_batch)    

    def load_grasp_label(self, file_path):
        """Returns a list of sorted (ascending on 'x') grasp labels
        from <file_path>.
        """
        grasp_list = []
        with open(file_path, 'r') as f:
            file = f.readlines()
            # grasp: 'x;y;theta;w;h'
            for grasp in file:
                # remove '\n' from string
                grasp = grasp[:-1]
                label = grasp.split(';')
                label = self.noramlize_grasp(label)
                grasp_list.append(label)

        grasp_list.sort(key=lambda x: x[0])
        return np.array(grasp_list)

    def get_grasp_label(self, grasp_list, metric='random'):
        """Returns the middle or a random grasp label for training."""
        # Selection method: 'x' median
        if metric == 'median':
            mid_idx = len(grasp_list) // 2
        # Selection method: random
        if metric == 'random':
            mid_idx = random.randint(0, len(grasp_list) - 1)

        return grasp_list[mid_idx]

    def noramlize_grasp(self, label):
        """Returns normalize grasping labels."""
        norm_label = []
        for i, value in enumerate(label):
            if i == 4:
                # Height
                norm_label.append(float(value) / 100)
            elif i == 2:
                # Theta
                norm_label.append((float(value) + 90) / 180)
            elif i == 3:
                # Width
                norm_label.append(float(value) / 1024)
            else:
                # Coordinates
                norm_label.append(float(value) / 1024)

        return norm_label


def rotate_grasp_label(grasp_list, degrees):
    # grasp_list.shape == (n, 5)
    # x, y, theta, w, h
    new_grasp_list =[]
    for grasp in grasp_list:
        x = grasp[0] * 1024
        y = grasp[1] * 1024

        angle = np.deg2rad(-degrees)
        R = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle),  np.cos(angle)]])
        o = np.atleast_2d((1024 // 2, 1024 // 2))
        p = np.atleast_2d((x, y))
        
        coords = np.squeeze((R @ (p.T-o.T) + o.T).T)
        if degrees == 0 or degrees == 180:
            theta = grasp[2] * 180 - 90
        elif degrees == 90 or degrees == -90:
            if grasp[2] <= 0.5 :
                theta = grasp[2] * 180
            elif grasp[2] > 0.5:
                theta = grasp[2] * 180 - 180
        
        w = grasp[3]
        h = grasp[4]

        new_grasp_list.append([coords[0] / 1024, coords[1] / 1024, (theta + 90) / 180, w, h])

    return np.array(new_grasp_list)


def crop_jitter_resize_img(img, ratio, jitter_x, jitter_y):
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


def crop_jitter_resize_label(grasp_label, ratio, jitter_x, jitter_y):
    grasp_label[:, 0] = (grasp_label[:, 0] - ((1-ratio)/2) - jitter_x) / ratio
    grasp_label[:, 1] = (grasp_label[:, 1] - ((1-ratio)/2) - jitter_y) / ratio
    grasp_label[:, 3] = grasp_label[:, 3] / ratio
    grasp_label[:, 4] = grasp_label[:, 4] / ratio

    return grasp_label


class DataLoader:
    def __init__(self):
        pass
    
    def load_rgbd(self):
        """Returns one-by-one all RGB images in the <dataset> folder."""
        for img_path in glob.glob('%s/*/*RGB.png' % params.DATA_PATH):
            # Get image subdirectory name ("left" / "right")
            img_dir_str = img_path.split('\\')[-2]
            # Get image id (e.g. 0_4e4a043d8c8cee6afad30cd586639ed2)
            img_path_str = img_path.split('\\')[-1]
            img_id = img_path_str[:-8]
            # Open RGB image with PIL
            img_rgb = Image.open(img_path)
            img_rgb = img_rgb.resize((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
            img_rgb = np.array(img_rgb)
            # Open depth image with PIL
            img_d = imread(os.path.join(params.DATA_PATH, img_dir_str, img_id + '_perfect_depth.tiff'))
            img_d = resize(img_d, (params.OUTPUT_SIZE, params.OUTPUT_SIZE), preserve_range=True).astype(img_d.dtype)
            img_d = np.array(img_d)

            yield (self.process(img_rgb, img_d), img_id)
    
    def process(self, rgb, d):
        """Returns rgbd image with correct format for inputing to model."""
        d = np.expand_dims(d, 2)
        img = np.concatenate((rgb, d), axis=2)
        img = np.moveaxis(img, -1, 0)
        img = np.expand_dims(img, 0)
        img = torch.tensor(img, dtype=torch.float32).to(params.DEVICE)

        return img


def jacquard_sin_loader(img_id):
    """Returns 'sin' ground truth map for images take from the Jaquard Dataset.
    
    Code referenced from @author: Sulabh Kumra - https://github.com/skumra/robotic-grasping"""
    # Get ground-truth path from img_id
    path = glob.glob('%s/*/%s_grasps.txt' % (params.DATA_PATH, img_id))[0]
    # Load all grasp rectangles from .txt file
    bbs = grasp.GraspRectangles.load_from_jacquard_file(path, scale=params.OUTPUT_SIZE / 1024.0)
    # Convert grasp rectangles into one single 'cos' map
    _, ang_img, _ = bbs.draw((params.OUTPUT_SIZE, params.OUTPUT_SIZE))
    sin = numpy_to_torch(np.sin(2 * ang_img))

    return sin


def numpy_to_torch(s):
    if len(s.shape) == 2:
        return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
    else:
        return torch.from_numpy(s.astype(np.float32))
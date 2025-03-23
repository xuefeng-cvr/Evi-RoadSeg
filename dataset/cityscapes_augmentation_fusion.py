import os
import numpy as np
import scipy.io as sio
import scipy.misc as m
import torch
from PIL import Image
from torch.utils import data
# from mypath import Path
from torchvision import transforms
import cv2
import glob
import json
import sys

from model.sne_model import SNE

from dataset import custom_transforms as tr
from scipy.ndimage.morphology import distance_transform_edt
from dataset.augmentation_cityscapes import add_tree_shadow, geometry_depth_aug
import random
import matplotlib.pyplot as plt


def Disp2depth(fx, baseline, disp, disp_path=''):
    # fx = 2268.36
    # baseline = 0.222126
    delta = 256
    if(disp is None):
        print("error!!! \n")
        print(disp_path)
        print("\n")
    try:
        disp_mask = disp > 0
    except:
        print("error!!! \n")
        print(disp_path)
        print("\n")
    depth = disp.astype(np.float32)
    depth[disp_mask] = (depth[disp_mask] - 1) / delta
    disp_mask = depth > 0
    depth[disp_mask] = fx * baseline / depth[disp_mask]
    return depth


def read_calib_file(filepath):
    with open(filepath, 'r') as f:
        calib_info = json.load(f)
        baseline = calib_info['extrinsic']['baseline']
        fx = calib_info['intrinsic']['fx']
        fy = calib_info['intrinsic']['fy']
        u0 = calib_info['intrinsic']['u0']
        v0 = calib_info['intrinsic']['v0']
    return baseline, fx, fy, u0, v0

def cityscapes_augmentation(img, depth_image, label, shadows, cityscapes_cars, instance_bottom_depth, cityscapes_root):
    img_aug, depth_aug, label_aug = geometry_depth_aug(img, depth_image, label, cityscapes_cars, instance_bottom_depth, cityscapes_root)
    # shadow augmentation
    random_shadow = np.random.randint(0, 11)
    shadow_path = shadows[random_shadow]
    shadow = cv2.imread(shadow_path, 0)
    img_aug = add_tree_shadow(img_aug, label_aug, shadow)
    return img_aug, depth_aug, label_aug


class CityscapesAugmentationFusionSegmentation(data.Dataset):
    NUM_CLASSES = 2

    def __init__(self, args, root='./data/Free/', split='train'):

        self.root = root
        self.split = split
        self.args = args
        self.images = {}
        self.disparities = {}
        self.calibs = {}
        self.labels = {}

        self.images_base = os.path.join(self.root, 'leftImg8bit_trainvaltest/leftImg8bit', self.split)
        self.disparities_base = os.path.join(self.root, 'disparity_trainvaltest/disparity', self.split)
        self.calibs_base = os.path.join(self.root, 'camera_trainvaltest/camera', self.split)
        self.annotations_base = os.path.join(self.root, 'gtFine_trainvaltest/gtFine', self.split)

        self.images[split] = []
        self.images[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')
        self.images[split].sort()

        self.disparities[split] = []
        self.disparities[split] = self.recursive_glob(rootdir=self.disparities_base, suffix='.png')
        self.disparities[split].sort()

        self.calibs[split] = []
        self.calibs[split] = self.recursive_glob(rootdir=self.calibs_base, suffix='.json')
        self.calibs[split].sort()

        self.labels[split] = []
        self.labels[split] = self.recursive_glob(rootdir=self.annotations_base, suffix='_labelIds.png')
        self.labels[split].sort()
        
        if split == 'train':
            # shadow
            self.shadow_folder = os.path.join(self.root, 'shadow')
            self.shadows = glob.glob(os.path.join(self.shadow_folder, '*.png'))
            self.shadows.sort()

            # cityscapes geometry
            self.cityscapes_root = args.cityscapes_path
            self.cityscapes_images_base = os.path.join(self.cityscapes_root, 'leftImg8bit_trainvaltest/leftImg8bit', 'train')
            self.cityscapes_disparities_base = os.path.join(self.cityscapes_root, 'disparity_trainvaltest/disparity', 'train')
            self.cityscapes_calibs_base = os.path.join(self.cityscapes_root, 'camera_trainvaltest/camera', 'train')
            self.cityscapes_annotations_base = os.path.join(self.cityscapes_root, 'gtFine_trainvaltest/gtFine', 'train')

            self.cityscapes_images = self.recursive_glob(rootdir=self.cityscapes_images_base, suffix='.png')
            self.cityscapes_images.sort()

            self.cityscapes_disparities = self.recursive_glob(rootdir=self.cityscapes_disparities_base, suffix='.png')
            self.cityscapes_disparities.sort()

            self.cityscapes_calibs = self.recursive_glob(rootdir=self.cityscapes_calibs_base, suffix='.json')
            self.cityscapes_calibs.sort()

            self.cityscapes_semantic = self.recursive_glob(rootdir=self.cityscapes_annotations_base, suffix='_labelIds.png')
            self.cityscapes_semantic.sort()

            self.cityscapes_labels = self.recursive_glob(rootdir=self.cityscapes_annotations_base, suffix='_instanceIds.png')
            self.cityscapes_labels.sort()

            self.cityscapes_cars = []
            for filename in os.listdir('RGBD_DA_files/ExObj_img/'):
                filename = os.path.join('RGBD_DA_files/ExObj_img/',filename)
                self.cityscapes_cars.append(filename)
            self.cityscapes_cars.sort()

            self.instance_bottom_depth = []
            with open('RGBD_DA_files/ExObj_depth.txt','r') as f:
                for line in f:
                    self.instance_bottom_depth.append(float(line))

        self.sne_model = SNE(crop_top=False)

        self.ignore_index = 255

        if not self.images[split]:
            raise Exception("No RGB images for split=[%s] found in %s" % (split, self.images_base))
        if not self.disparities[split]:
            raise Exception("No depth images for split=[%s] found in %s" % (split, self.disparities_base))

        print("Found %d %s RGB images" % (len(self.images[split]), split))
        print("Found %d %s disparity images" % (len(self.disparities[split]), split))

    def __len__(self):
        return len(self.images[self.split])

    def __getitem__(self, index):

        img_path = self.images[self.split][index].rstrip()
        disp_path = self.disparities[self.split][index].rstrip()
        calib_path = self.calibs[self.split][index].rstrip()
        lbl_path = self.labels[self.split][index].rstrip()

        useDir = "/".join(img_path.split('/')[:-2])
        name = img_path.split('/')[-1]

        label_image = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        oriHeight, oriWidth = label_image.shape
        label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
        label[label_image == 7] = 1
        label[label_image != 7] = 0

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        disp_image = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH)
        baseline, fx, fy, u0, v0 = read_calib_file(calib_path)
        depth = Disp2depth(fx, baseline, disp_image, disp_path)

        # augmentation
        if self.split == 'train':
            
            img_aug, depth_aug, label_aug = cityscapes_augmentation(img, depth, label, self.shadows, self.cityscapes_cars, self.instance_bottom_depth, self.cityscapes_root)

            for i in range(4):
                # 2 / 3 augmentation iters
                if random.random() < 0.5:
                    img_aug, depth_aug, label_aug = cityscapes_augmentation(img, depth, label, self.shadows, self.cityscapes_cars, self.instance_bottom_depth, self.cityscapes_root)

        else:
            img_aug, depth_aug, label_aug = img, depth, label

        _img = Image.fromarray(img_aug)
        _depth = Image.fromarray(depth_aug)
        _target = Image.fromarray(label_aug)

        sample = {'image': _img, 'depth': _depth, 'label': _target}

        if self.split == 'train':
            sample = self.transform_tr(sample)
        elif self.split == 'val':
            sample = self.transform_val(sample)
        elif self.split == 'test':
            sample = self.transform_ts(sample)
        else:
            sample = self.transform_ts(sample)
        
        depth_image = np.array(sample['depth'])

        calib = np.array([[fx, 0, u0],
                        [0, fy, v0],
                        [0, 0, 1]])
        camParam = torch.tensor(calib, dtype=torch.float32)
        normal = self.sne_model(torch.tensor(depth_image.astype(np.float32)), camParam)
        normal = normal.cpu().numpy()
        normal = np.transpose(normal, [1, 2, 0])
        normal = cv2.resize(normal, (self.args.crop_width, self.args.crop_height))
        # normal = np.transpose(normal, [2, 0, 1])

        normal = transforms.ToTensor()(normal)

        sample['depth'] = normal

        sample['label'] = np.array(sample['label'])
        sample['label'] = torch.from_numpy(sample['label']).long()

        sample['oriHeight'] = oriHeight
        sample['oriWidth'] = oriWidth

        return sample

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.Resize_without_depth(size=(self.args.crop_width, self.args.crop_height)),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.Resize_without_depth(size=(self.args.crop_width, self.args.crop_height)),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.Resize_without_depth(size=(self.args.crop_width, self.args.crop_height)),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)


def get_lostandfound_labels():
    return np.array([
        [0, 0, 0],
        [128, 64, 128],
        [0, 0, 142]])

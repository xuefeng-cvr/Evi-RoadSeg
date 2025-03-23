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
from model.sne_model import SNE
from dataset import custom_transforms as tr
from scipy.ndimage.morphology import distance_transform_edt
from utils import get_label_info
from dataset.augmentation_kitti import add_tree_shadow, geometry_depth_aug
import random
import matplotlib.pyplot as plt

class kittiCalibInfo():
    """
    Read calibration files in the kitti dataset,
    we need to use the intrinsic parameter of the cam2
    """
    def __init__(self, filepath):
        """
        Args:
            filepath ([str]): calibration file path (AAA.txt)
        """
        self.data = self._load_calib(filepath)

    def get_cam_param(self):
        """
        Returns:
            [numpy.array]: intrinsic parameter of the cam2
        """
        return self.data['P2']

    def _load_calib(self, filepath):
        rawdata = self._read_calib_file(filepath)
        data = {}
        P0 = np.reshape(rawdata['P0'], (3,4))
        P1 = np.reshape(rawdata['P1'], (3,4))
        P2 = np.reshape(rawdata['P2'], (3,4))
        P3 = np.reshape(rawdata['P3'], (3,4))
        R0_rect = np.reshape(rawdata['R0_rect'], (3,3))
        Tr_velo_to_cam = np.reshape(rawdata['Tr_velo_to_cam'], (3,4))

        data['P0'] = P0
        data['P1'] = P1
        data['P2'] = P2
        data['P3'] = P3
        data['R0_rect'] = R0_rect
        data['Tr_velo_to_cam'] = Tr_velo_to_cam

        return data

    def _read_calib_file(self, filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}

        with open(filepath, 'r') as f:
            for line in f.readlines():
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

def read_calib_file(filepath):
    with open(filepath, 'r') as f:
        calib_info = json.load(f)
        baseline = calib_info['extrinsic']['baseline']
        fx = calib_info['intrinsic']['fx']
        fy = calib_info['intrinsic']['fy']
        u0 = calib_info['intrinsic']['u0']
        v0 = calib_info['intrinsic']['v0']
    return baseline, fx, fy, u0, v0

def kitti_augmentation(img, depth_image, label, shadows, cityscapes_cars, instance_bottom_depth, cityscapes_root):
    img_aug, depth_aug, label_aug = geometry_depth_aug(img, depth_image, label, cityscapes_cars, instance_bottom_depth, cityscapes_root)
    random_shadow = np.random.randint(0, 11)
    shadow_path = shadows[random_shadow]
    shadow = cv2.imread(shadow_path, 0)
    img_aug = add_tree_shadow(img_aug, label_aug, shadow)
    return img_aug, depth_aug, label_aug


class KittiBevSneAugmentationFusionSegmentation(data.Dataset):

    def __init__(self, args, root='./data/Free/', split='train'):

        self.root = root
        self.split = split
        self.args = args
        self.images = {}
        self.disparities = {}
        self.labels = {}
        self.calibs = {}

        self.images_base = os.path.join(self.root, self.split, 'image_2')
        self.disparities_base = os.path.join(self.root, self.split, 'depth_u16')
        self.annotations_base = os.path.join(self.root, self.split, 'gt_image_2')
        self.calib_base = os.path.join(self.root, self.split, 'calib')

        self.images[split] = []
        self.images[split].extend(glob.glob(os.path.join(self.images_base, '*.png')))
        self.images[split].sort()

        self.disparities[split] = []
        self.disparities[split].extend(glob.glob(os.path.join(self.disparities_base, '*.png')))
        self.disparities[split].sort()

        self.labels[split] = []
        self.labels[split].extend(glob.glob(os.path.join(self.annotations_base, '*.png')))
        self.labels[split].sort()

        self.calibs[split] = []
        self.calibs[split].extend(glob.glob(os.path.join(self.calib_base, '*.txt')))
        self.calibs[split].sort()

        if self.split == 'training' or self.split == 'training_all' \
            or self.split == 'training_1' or self.split == 'training_2':
            # shadow
            self.shadow_folder = os.path.join(self.root, 'shadow')
            self.shadows = glob.glob(os.path.join(self.shadow_folder, '*.png'))
            self.shadows.sort()

            # cityscapes
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

        self.sne_model = SNE()

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

        useDir = "/".join(img_path.split('/')[:-2])
        name = img_path.split('/')[-1]
        lbl_path = os.path.join(useDir, 'gt_image_2', name[:-10]+'road_'+name[-10:])

        label_image = cv2.cvtColor(cv2.imread(lbl_path), cv2.COLOR_BGR2RGB)
        oriHeight, oriWidth, _ = label_image.shape
        label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
        label[label_image[:, :, 2] > 0] = 1

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH)
        depth_image = depth_image.astype(np.float32)
        depth_image = depth_image / 1000

        # augmentation
        if self.split == 'training' or self.split == 'training_all' \
                or self.split == 'training_1' or self.split == 'training_2':
            img_aug, depth_aug, label_aug = kitti_augmentation(img, depth_image, label, self.shadows, self.cityscapes_cars, self.instance_bottom_depth, self.cityscapes_root)
            for i in range(4):
                # 2 / 3 augmentation iters
                if random.random() < 0.5:
                    img_aug, depth_aug, label_aug = kitti_augmentation(img, depth_image, label, self.shadows, self.cityscapes_cars, self.instance_bottom_depth, self.cityscapes_root)
        else:
            img_aug, depth_aug, label_aug = img, depth_image, label

        _img = Image.fromarray(img_aug)
        _depth = Image.fromarray(depth_aug)

        _target = Image.fromarray(label_aug)

        sample = {'image': _img, 'depth': _depth, 'label': _target}

        if self.split == 'training' or self.split == 'training_all' \
            or self.split == 'training_1' or self.split == 'training_2':
            sample = self.transform_tr(sample)
        elif self.split == 'validating':
            sample = self.transform_val(sample)
        elif self.split == 'testing':
            sample = self.transform_ts(sample)
        else:
            sample = self.transform_ts(sample)

        depth_image = np.array(sample['depth'])

        calib = kittiCalibInfo(calib_path)
        camParam = torch.tensor(calib.get_cam_param(), dtype=torch.float32)
        # normal = self.sne_model(torch.tensor(depth_image.astype(np.float32) / 1000), camParam)
        normal = self.sne_model(torch.tensor(depth_image.astype(np.float32)), camParam)
        normal = normal.cpu().numpy()
        normal = np.transpose(normal, [1, 2, 0])
        normal = cv2.resize(normal, (self.args.crop_width, self.args.crop_height))

        normal = transforms.ToTensor()(normal)

        sample['depth'] = normal

        sample['label'] = np.array(sample['label'])
        sample['label'] = torch.from_numpy(sample['label']).long()

        sample['oriHeight'] = oriHeight
        sample['oriWidth'] = oriWidth
        sample['calib_path'] = calib_path

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

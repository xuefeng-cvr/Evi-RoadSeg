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


from model.sne_model import SNE

from dataset import custom_transforms as tr
from dataset.augmentation_r2d import add_tree_shadow, geometry_depth_aug
import random


class orfdCalibInfo():
    """
    Read calibration files in the ORFD dataset,
    we need to use the intrinsic parameter
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
            [numpy.array]: intrinsic parameter 
        """
        return self.data['K']

    def _load_calib(self, filepath):
        rawdata = self._read_calib_file(filepath)
        data = {}
        K = np.reshape(rawdata['cam_K'], (3,3))
        data['K'] = K
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

def r2d_augmentation(img, depth_image, label, shadows, cityscapes_cars, instance_bottom_depth, cityscapes_root):
    img_aug, depth_aug, label_aug = geometry_depth_aug(img, depth_image, label, cityscapes_cars, instance_bottom_depth, cityscapes_root)
    # shadow augmentation
    random_shadow = np.random.randint(0, 11)
    shadow_path = shadows[random_shadow]
    shadow = cv2.imread(shadow_path, 0)
    img_aug = add_tree_shadow(img_aug, label_aug, shadow)
    return img_aug, depth_aug, label_aug


class R2DAugmentationFusionSegmentation(data.Dataset):

    def __init__(self, args, root='./data/Free/', split='training'):

        self.root = root
        self.split = split
        self.args = args
        self.images = {}
        self.depths = {}
        self.labels = {}

        self.image_base = os.path.join(self.root, self.split, '*', 'rgb')
        self.depth_base = os.path.join(self.root, self.split, '*', 'depth')
        self.label_base = os.path.join(self.root, self.split, '*', 'label')

        self.images[split] = []
        self.images[split].extend(glob.glob(os.path.join(self.image_base, '*.jpg')))
        self.images[split].sort()

        self.depths[split] = []
        self.depths[split].extend(glob.glob(os.path.join(self.depth_base, '*.png')))
        self.depths[split].sort()

        self.labels[split] = []
        self.labels[split].extend(glob.glob(os.path.join(self.label_base, '*.png')))
        self.labels[split].sort()
        
        if split == 'training':
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


        if not self.images[split]:
            raise Exception("No RGB images for split=[%s] found in %s" % (split, self.image_base))
        if not self.depths[split]:
            raise Exception("No depth images for split=[%s] found in %s" % (split, self.depth_base))

        print("Found %d %s RGB images" % (len(self.images[split]), split))
        print("Found %d %s depth images" % (len(self.depths[split]), split))

    def __len__(self):
        return len(self.images[self.split])

    def __getitem__(self, index):

        img_path = self.images[self.split][index].rstrip()
        depth_path = self.depths[self.split][index].rstrip()
        lbl_path = self.labels[self.split][index].rstrip()
        name = img_path.split('/')[-3] + '_' + img_path.split('/')[-1]


        label_image = cv2.imread(lbl_path)
        oriHeight, oriWidth, _  = label_image.shape
        label = np.zeros((oriHeight, oriWidth), dtype=np.uint8)
        label[label_image[:, :, 2] == 7] = 1
        
        img = img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        depth_image = cv2.imread(depth_path).astype('float32')
        normalized = (depth_image[:, :, 2] + depth_image[:, :, 1] * 256 + depth_image[:, :, 0] * 256 * 256) / (256 * 256 * 256 - 1)
        depth_img = 1000 * normalized
        calib = np.identity(3)

        # augmentation
        if self.split == 'training':
            img_aug, depth_aug, label_aug = r2d_augmentation(img, depth_img, label, self.shadows, self.cityscapes_cars, self.instance_bottom_depth, self.cityscapes_root)

            for i in range(4):
                # 2 / 3 augmentation iters
                if random.random() < 0.5:
                    img_aug, depth_aug, label_aug = r2d_augmentation(img, depth_img, label, self.shadows, self.cityscapes_cars, self.instance_bottom_depth, self.cityscapes_root)
        else:
            img_aug, depth_aug, label_aug = img, depth_img, label

        _img = Image.fromarray(img_aug)
        _depth = Image.fromarray(depth_aug)
        _target = Image.fromarray(label_aug)

        sample = {'image': _img, 'depth': _depth, 'label': _target}

        if self.split == 'training':
            sample = self.transform_tr(sample)
        elif self.split == 'validation':
            sample = self.transform_val(sample)
        elif self.split == 'testing':
            sample = self.transform_ts(sample)
        else:
            sample = self.transform_ts(sample)
        
        depth_image = np.array(sample['depth'])

        camParam = torch.tensor(calib, dtype=torch.float32)
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

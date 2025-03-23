import numbers
import torch
import random
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import cv2


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        depth = np.array(depth).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        # # mean and std for original depth images
        # mean_depth = 0.12176
        # std_depth = 0.09752

        # depth /= 255.0
        # depth -= mean_depth
        # depth /= std_depth

        return {'image': img,
                'depth': depth,
                'label': mask}


class ToTensor(object):
    """Convert Image object in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        depth = np.array(depth).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        depth = torch.from_numpy(depth).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'depth': depth,
                'label': mask}


class ColorJitter(object):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4):
        self.brightness = [max(1 - brightness, 0), 1 + brightness]
        self.contrast = [max(1 - contrast, 0), 1 + contrast]
        self.saturation = [max(1 - saturation, 0), 1 + saturation]

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        img = ImageEnhance.Brightness(img).enhance(r_brightness)
        img = ImageEnhance.Contrast(img).enhance(r_contrast)
        img = ImageEnhance.Color(img).enhance(r_saturation)
        return {'image': img,
                'depth': depth,
                'label': mask}


class CropBlackArea(object):
    """
    crop black area for depth image
    """
    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        width, height = img.size
        left = 140
        top = 30
        right = 2030
        bottom = 900
        # crop
        img = img.crop((left, top, right, bottom))
        depth = depth.crop((left, top, right, bottom))
        mask = mask.crop((left, top, right, bottom))
        # resize
        img = img.resize((width,height), Image.BILINEAR)
        depth = depth.resize((width,height), Image.BILINEAR)
        mask = mask.resize((width,height), Image.NEAREST)

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            # depth = np.flip(depth, 2)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'depth': depth,
                'label': mask}


class HorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        depth = depth.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomGaussianBlur(object):
    def __init__(self, radius=1):
        self.radius = radius

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=self.radius*random.random()))

        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomGaussianNoise(object):
    def __init__(self, mean=0, sigma=10):
        self.mean = mean
        self.sigma = sigma

    def gaussianNoisy(self, im, mean=0, sigma=10):
        noise = np.random.normal(mean, sigma, len(im))
        im = im + noise

        im = np.clip(im, 0, 255)
        return im

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        if random.random() < 0.5:
            # 将图像转化成数组
            img = np.asarray(img)
            img = img.astype(np.int)
            width, height = img.shape[:2]
            img_r = self.gaussianNoisy(img[:, :, 0].flatten(), self.mean, self.sigma)
            img_g = self.gaussianNoisy(img[:, :, 1].flatten(), self.mean, self.sigma)
            img_b = self.gaussianNoisy(img[:, :, 2].flatten(), self.mean, self.sigma)
            img[:, :, 0] = img_r.reshape([width, height])
            img[:, :, 1] = img_g.reshape([width, height])
            img[:, :, 2] = img_b.reshape([width, height])
            img = Image.fromarray(np.uint8(img))
        return {'image': img,
                'depth': depth,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        depth = depth.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            depth = ImageOps.expand(depth, border=(0, 0, padw, padh), fill=0)  # depth多余的部分填0
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        depth = depth.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'depth': depth,
                'label': mask}

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding
 
    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
 
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            depth = ImageOps.expand(depth, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size # target size
        if w == tw and h == th:
            return {'image': img,
                    'depth': depth,
                    'label': mask}
        if w < tw or h < th:
            img = img.resize((tw, th), Image.BILINEAR)
            depth = depth.resize((tw, th), Image.BILINEAR)
            mask = mask.resize((tw, th), Image.NEAREST)
            return {'image': img,
                    'depth': depth,
                    'label': mask}
 
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = img.crop((x1, y1, x1 + tw, y1 + th))
        # depth = depth.crop((x1, y1, x1 + tw, y1 + th))
        depth = depth[:, y1:y1+th, x1:x1+tw]
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))
 
        return {'image': img, 
                'depth': depth, 
                'label': mask}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        depth = depth.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        depth = depth.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'depth': depth,
                'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        assert img.size == depth.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        depth = depth.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'depth': depth,
                'label': mask}


class Resize(object):
    def __init__(self, size):
        self.size = size# size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        assert img.size == depth.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        depth = depth.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'depth': depth,
                'label': mask}


class Resize_without_depth(object):
    def __init__(self, size):
        self.size = size# size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        # assert img.size == depth.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        # depth = depth.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'depth': depth,
                'label': mask}

class Resize_with_depth(object):
    def __init__(self, size):
        self.size = size# size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        # assert img.size == depth.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        depth = np.transpose(depth, [1, 2, 0])
        depth = cv2.resize(depth, (self.size[0], self.size[1]))
        depth = np.transpose(depth, [2, 0, 1])
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'depth': depth,
                'label': mask}

class Resize_with_depth_without_label(object):
    def __init__(self, size):
        self.size = size# size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        # assert img.size == depth.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        depth = np.transpose(depth, [1, 2, 0])
        depth = cv2.resize(depth, (self.size[0], self.size[1]))
        depth = np.transpose(depth, [2, 0, 1])
        # mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'depth': depth,
                'label': mask}

class  resize_without_depth_label(object):
    def __init__(self, size):
        self.size = size# size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        assert img.size == depth.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        # depth = depth.resize(self.size, Image.BILINEAR)
        # mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'depth': depth,
                'label': mask}


class Resize_without_label(object):
    def __init__(self, size):
        self.size = size# size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        depth = sample['depth']
        mask = sample['label']

        assert img.size == depth.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        depth = depth.resize(self.size, Image.BILINEAR)
        # mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'depth': depth,
                'label': mask}


class Relabel(object):
    def __init__(self, olabel, nlabel):  # change trainid label from olabel to nlabel
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        # assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor,
        #                                                            torch.ByteTensor)), 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor
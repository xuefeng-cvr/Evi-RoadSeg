import torch.nn as nn
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import random
import numbers
import torchvision
import skimage.measure
import os
from tqdm import tqdm
from scipy.ndimage.morphology import distance_transform_edt


def mask_to_binary_edges(mask, radius):
    """
    Converts a segmentation mask (K,H,W) to a binary edgemap (H,W)
    """

    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((1, 1), (1, 1)), mode='constant', constant_values=0)

    dist_in = distance_transform_edt(mask_pad)
    dist_in = dist_in[1:-1, 1:-1]
    dist_in[dist_in > radius] = 0

    dist_out = distance_transform_edt(1.0 - mask_pad)
    dist_out = dist_out[1:-1, 1:-1]
    dist_out[dist_out > radius * 2] = 0

    edgemap = dist_in + dist_out

    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap


def calculate_weigths_labels(dataloader, num_classes, save_path):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for sample in tqdm_batch:
        y = sample['label']
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)
    classes_weights_path = os.path.join(save_path, 'classes_weights.npy')
    np.save(classes_weights_path, ret)

    return ret


def calculate_weights_positions(label):
    # Calculate weights of all positions
    b, h, w = label.shape
    weight_position = np.zeros((b, h, w))
    for i in range(b):
        sum_r = np.sum(label[i, :, :], 1)
        sum_r[sum_r < 20] = 20
        med = np.median(np.unique(sum_r))
        weight = np.tile(sum_r.reshape(h, 1), w)
        weight[weight == 20] = med
        weight_position[i, :, :] = med / weight
    weight_position = torch.from_numpy(weight_position.astype(np.float32))
    return weight_position


def calculate_weights_positions_crop_nonroad(label):
    # Calculate weights of all positions
    b, h, w = label.shape
    weight_position = np.zeros((b, h, w))
    for i in range(b):
        sum_r = np.sum(label[i, :, :], 1)
        sum_r[sum_r < 20] = 20
        med = np.median(np.unique(sum_r))
        weight = np.tile(sum_r.reshape(h, 1), w)
        weight[weight == 20] = med

        label_binary = (label[i, :, :] > 0).astype(np.uint8)
        edgemap = mask_to_binary_edges(label_binary, 8)
        mask_nonroad = (edgemap + label_binary) == 0
        weight[mask_nonroad] = med

        weight_position[i, :, :] = med / weight
    weight_position = torch.from_numpy(weight_position.astype(np.float32))
    return weight_position


def calculate_boundary_aware_weights(label, output):
    # Calculate boundary aware weights
    b, h, w = label.shape
    label = label.astype(np.uint8)
    TN = ((label==0) * (output==0)).astype(np.uint8)
    FN = ((label==1) * (output==0)).astype(np.uint8)
    TP = ((label==1) * (output==1)).astype(np.uint8)
    FP = ((label==0) * (output==1)).astype(np.uint8)
    boundary_aware_weight = np.zeros((b, h,w))
    for i in range(b):
        w1 = cv2.distanceTransform((1-TN[i,:,:]).squeeze(), cv2.DIST_L2, cv2.DIST_MASK_3)
        w1 = w1 / w1.max()
        w1 = (1 - w1)*FP[i,:,:]
        w2 = cv2.distanceTransform((1-TP[i,:,:]).squeeze(), cv2.DIST_L2, cv2.DIST_MASK_3)
        w2 = w2 / w2.max()
        w2 = (1 - w2)*FN[i,:,:]
        boundary_aware_weight[i,:,:] = w1 + w2
    boundary_aware_weight = torch.from_numpy(boundary_aware_weight.astype(np.float32))
    return boundary_aware_weight


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power

	"""
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr * (1 - iter / max_iter) ** power
    optimizer.param_groups[0]['lr'] = lr
    return lr


def poly_lr_scheduler_2_group(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power

	"""
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr * (1 - iter / max_iter) ** power
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = lr * 0.1
    return lr


def get_label_info(csv_path):
    # return label -> {label_name: [r_value, g_value, b_value, ...}
    ann = pd.read_csv(csv_path)
    label = {}
    for iter, row in ann.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        class_11 = row['class_11']
        label[label_name] = [int(r), int(g), int(b), class_11]
    return label


def one_hot_it(label, label_info):
    # return semantic_map -> [H, W]
    semantic_map = np.zeros(label.shape[:-1])
    for index, info in enumerate(label_info):
        color = label_info[info]
        # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map[class_map] = index
    # semantic_map.append(class_map)
    # semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def one_hot_it_v11(label, label_info):
    # return semantic_map -> [H, W, class_num]
    semantic_map = np.zeros(label.shape[:-1])
    # from 0 to 11, and 11 means void
    class_index = 0
    for index, info in enumerate(label_info):
        color = label_info[info][:3]
        class_11 = label_info[info][3]
        if class_11 == 1:
            # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            # semantic_map[class_map] = index
            semantic_map[class_map] = class_index
            class_index += 1
        else:
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            semantic_map[class_map] = 11
    return semantic_map


def one_hot_it_v11_dice(label, label_info):
    # return semantic_map -> [H, W, class_num]
    semantic_map = []
    void = np.zeros(label.shape[:2])
    for index, info in enumerate(label_info):
        color = label_info[info][:3]
        class_11 = label_info[info][3]
        if class_11 == 1:
            # colour_map = np.full((label.shape[0], label.shape[1], label.shape[2]), colour, dtype=int)
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            # semantic_map[class_map] = index
            semantic_map.append(class_map)
        else:
            equality = np.equal(label, color)
            class_map = np.all(equality, axis=-1)
            void[class_map] = 1
    semantic_map.append(void)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float)
    return semantic_map


def one_hot(label, label_info):
    # return semantic_map -> [H, W, class_num]
    semantic_map = []
    void = np.zeros(label.shape[:2])

    for index, info in enumerate(label_info):
        equality = np.equal(label, index)
        void[equality] = 1
        # semantic_map[class_map] = index
        semantic_map.append(void)
        void = np.zeros(label.shape[:2])
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float)
    return semantic_map


def reverse_one_hot(image):
    """
	Transform a 2D array in one-hot format (depth is num_classes),
	to a 2D array with only 1 channel, where each pixel value is
	the classified class key.

	# Arguments
		image: The one-hot format image

	# Returns
		A 2D array with the same width and height as the input, but
		with a depth size of 1, where each pixel value is the classified
		class key.
	"""
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index
    if image.dim() == 3:
        image = image.permute(1, 2, 0)
    else:
        image = image.permute(0, 2, 3, 1)
    x = torch.argmax(image, dim=-1)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    label_values = [label_values[key][:3] for key in label_values if label_values[key][3] == 1]
    label_values.append([0, 0, 0])
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def compute_global_accuracy(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    total = len(label)
    count = 0.0
    # for i in range(total):
    #     if pred[i] == label[i]:
    #         count = count + 1.0
    count = np.sum(pred == label)
    return float(count) / float(total)


def fast_hist(a, b, n):
    '''
	a and b are predict and mask respectively
	n is the number of classes
	'''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    epsilon = 1e-5
    return (np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


def cal_cdi(pred, label):
    ground = skimage.measure.label(label == 2, connectivity=1)
    predict = skimage.measure.label(pred == 2, connectivity=1)

    inter = ground * (predict != 0)
    cdi = 0
    ti = np.max(ground)

    for i in range(1, np.max(ground) + 1):
        inter_rate = np.sum(inter == i) / np.sum(ground == i)
        if inter_rate > 0.5:
            cdi = cdi + 1
    return cdi, ti


def cal_idi(pred, label):
    ground = skimage.measure.label(label == 2, connectivity=1)
    predict = skimage.measure.label(pred == 2, connectivity=1)

    inter = predict * (ground != 0)
    idi = 0

    for i in range(1, np.max(predict) + 1):
        inter_rate = np.sum(inter == i) / np.sum(predict == i)
        if inter_rate < 0.5:
            idi = idi + 1
    return idi


def cal_pdr(hist):
    epsilon = 1e-5
    return ((np.diag(hist) + epsilon) / (hist.sum(1) + epsilon))[-1]


def cal_pfp(hist):
    epsilon = 1e-5
    return (np.sum(hist[:-1, -1]) + 1e-5)/(np.sum(hist.sum(1)[:-1]) + 1e-5)


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

	Args:
		size (sequence or int): Desired output size of the crop. If size is an
			int instead of sequence like (h, w), a square crop (size, size) is
			made.
		padding (int or sequence, optional): Optional padding on each border
			of the image. Default is 0, i.e no padding. If a sequence of length
			4 is provided, it is used to pad left, top, right, bottom borders
			respectively.
		pad_if_needed (boolean): It will pad the image if smaller than the
			desired size to avoid raising an exception.
	"""

    def __init__(self, size, seed, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.seed = seed

    @staticmethod
    def get_params(img, output_size, seed):
        """Get parameters for ``crop`` for a random crop.

		Args:
			img (PIL Image): Image to be cropped.
			output_size (tuple): Expected output size of the crop.

		Returns:
			tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
		"""
        random.seed(seed)
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
		Args:
			img (PIL Image): Image to be cropped.

		Returns:
			PIL Image: Cropped image.
		"""
        if self.padding > 0:
            img = torchvision.transforms.functional.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = torchvision.transforms.functional.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = torchvision.transforms.functional.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size, self.seed)

        return torchvision.transforms.functional.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def cal_miou(miou_list, csv_path):
    # return label -> {label_name: [r_value, g_value, b_value, ...}
    ann = pd.read_csv(csv_path)
    miou_dict = {}
    cnt = 0
    for iter, row in ann.iterrows():
        label_name = row['name']
        class_11 = int(row['class_11'])
        if class_11 == 1:
            miou_dict[label_name] = miou_list[cnt]
            cnt += 1
    return miou_dict, np.mean(miou_list)


def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        # return 0, 0, 0, 0,
        return 0, 0, 0, 0, 0, 0, 0
    with np.errstate(divide='ignore',invalid='ignore'):
        globalacc = np.diag(conf_matrix).sum() / np.float(conf_matrix.sum())
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(np.float)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(np.float)
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(np.float)
        pre = classpre[1]
        recall = classrecall[1]
        iou = IU[1]
        F_score = 2*(recall*pre)/(recall+pre)
        fpr = conf_matrix[0, 1] / np.float(conf_matrix[0, 0] + conf_matrix[0, 1])
        fnr = conf_matrix[1, 0] / np.float(conf_matrix[1, 0] + conf_matrix[1, 1])
    return globalacc, pre, recall, F_score, iou, fpr, fnr


class OHEM_CrossEntroy_Loss(nn.Module):
    def __init__(self, threshold, keep_num):
        super(OHEM_CrossEntroy_Loss, self).__init__()
        self.threshold = threshold
        self.keep_num = keep_num
        self.loss_function = nn.CrossEntropyLoss(reduction='none')

    def forward(self, output, target):
        loss = self.loss_function(output, target).view(-1)
        loss, loss_index = torch.sort(loss, descending=True)
        threshold_in_keep_num = loss[self.keep_num]
        if threshold_in_keep_num > self.threshold:
            loss = loss[loss > self.threshold]
        else:
            loss = loss[:self.keep_num]
        return torch.mean(loss)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(
        group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group

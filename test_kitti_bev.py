import cv2
import argparse
from model.usnet import USNet
from bev.BirdsEyeView import BirdsEyeView
import os
import torch
import cv2
from model.sne_model import SNE
from PIL import Image
from torchvision import transforms
from dataset import custom_transforms as tr
import numpy as np
import time


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


def predict_on_image(model, img_path, depth_path, label_path, calib_path, args):
    # test Kitti
    composed_transforms = transforms.Compose([
        tr.Resize(size=(args.crop_width, args.crop_height)),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    sne_model = SNE()

    if args.use_sne:
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        # depth = Image.open(depth_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth_image = depth_image.astype(np.float32)
        depth_image = cv2.resize(depth_image, (w, h))
        depth = Image.fromarray(depth_image)
        label_image = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)
        label = np.zeros((h, w), dtype=np.uint8)
        label[label_image[:, :, 2] > 0] = 1
        label = Image.fromarray(label)
        w, h = image.size

        sample = {'image': image, 'depth': depth, 'label': label}
        sample = composed_transforms(sample)

        calib = kittiCalibInfo(calib_path)
        camParam = torch.tensor(calib.get_cam_param(), dtype=torch.float32)
        normal = sne_model(torch.tensor(depth_image.astype(np.float32) / 1000), camParam)
        normal = normal.cpu().numpy()
        normal = np.transpose(normal, [1, 2, 0])
        normal = transforms.ToTensor()(normal)

        sample['depth'] = normal
    else:
        image = Image.open(img_path).convert('RGB')
        w, h = image.size
        # depth = Image.open(depth_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        depth_image = depth_image.astype(np.float32)
        depth = Image.fromarray(depth_image)
        label_image = cv2.cvtColor(cv2.imread(label_path), cv2.COLOR_BGR2RGB)
        label = np.zeros((h, w), dtype=np.uint8)
        label[label_image[:, :, 2] > 0] = 1
        label = Image.fromarray(label)
        w, h = image.size

        sample = {'image': image, 'depth': depth, 'label': label}
        sample = composed_transforms(sample)
    
    # resize normal
    normal = normal.cpu().numpy()
    normal = np.transpose(normal, [1, 2, 0])
    normal = cv2.resize(normal, (args.crop_width, args.crop_height))
    normal = transforms.ToTensor()(normal)
    sample['depth'] = normal

    image = sample['image']
    depth = sample['depth']

    image = image.unsqueeze(0)
    depth = depth.unsqueeze(0)

    # predict
    with torch.no_grad():
        # draw probmap
        model.eval()
        evidence, evidence_a, alpha, alpha_a = model(image, depth)
        s = torch.sum(alpha_a, dim=1, keepdim=True)
        # s = torch.sum(evidence_a, dim=1, keepdim=True)
        b = alpha_a / (s.expand(alpha_a.shape))
        u = args.num_classes / s
        
        predict = b[:,1]
        predict = predict.view(args.crop_height, args.crop_width)
        predict = predict.detach().cpu().numpy()
        predict = np.array(predict)

        label = cv2.resize(np.uint8(label), (w, h), interpolation=cv2.INTER_NEAREST)
        predict = cv2.resize(predict, (w, h))

    return label, predict


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_sne', action='store_true', default=False, help='Whether to user sne')
    parser.add_argument('--image', action='store_true', default=False, help='predict on image')
    parser.add_argument('--video', action='store_true', default=False, help='predict on video')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='The path to the pretrained weights of model')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--num_classes', type=int, default=2, help='num of object classes (with void)')
    parser.add_argument('--data', type=str, default=None, help='Path to image or video for prediction')
    parser.add_argument('--crop_height', type=int, default=384, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1248, help='Width of cropped/resized input image to network')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='Path to save predict image')


    args = parser.parse_args(params)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = USNet(args.num_classes, args.context_path)
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    print('load model from %s ...' % args.checkpoint_path)
    model.module.load_state_dict(torch.load(args.checkpoint_path))
    print('Done!')

    timestart = time.time()

    image_folder = os.path.join(args.data, 'testing', 'image_2')
    depth_folder = os.path.join(args.data, 'testing', 'depth_u16')
    label_folder = os.path.join(args.data, 'testing', 'gt_image_2')
    calib_folder = os.path.join(args.data, 'testing', 'calib')

    imgs = os.listdir(image_folder)
    # BEV class
    bev = BirdsEyeView()
    for img in imgs:
        depth = img
        label = img[:-10]+'road_'+img[-10:]
        calib = img[:-4] + '.txt'
        img_path = os.path.join(image_folder, img)
        depth_path = os.path.join(depth_folder, depth)
        label_path = os.path.join(label_folder, label)
        calib_path = os.path.join(calib_folder, calib)
        label, predict = predict_on_image(model, img_path, depth_path, label_path, calib_path, args)
        
        # draw probmap
        result = np.floor(255*(predict - predict.min()) / (predict.max()-predict.min()))
        result = result.astype(np.uint8)

        # Update calibration for Birds Eye View
        bev.setup(calib_path)

        # Compute Birds Eye View
        result = bev.compute(result)

        save_dir = args.save_path
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_name = img.split('_')[0]+'_road_'+img.split('_')[1]
        save_path = os.path.join(save_dir, save_name)
        cv2.imwrite(save_path, np.uint8(result))

    timeend = time.time()

    print(timeend-timestart)

    # predict on video
    if args.video:
        pass

if __name__ == '__main__':
    params = [
        '--use_sne',
        '--image',
        '--data', '/data/KITTI/KITTI',
        '--context_path', 'resnet18',
        '--cuda', '0',
        '--num_classes', '2',
        '--checkpoint_path', './weight/kitti/best_model.pth',
        '--save_path', './result/kitti',
    ]
    main(params)

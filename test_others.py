from torch.utils.data import DataLoader
import argparse
from dataset.cityscapes_augmentation_fusion import CityscapesAugmentationFusionSegmentation
from dataset.orfd_augmentation_fusion import OrfdAugmentationFusionSegmentation
from dataset.r2d_augmentation_fusion import R2DAugmentationFusionSegmentation
import os
from model.usnet import USNet
import torch
import numpy as np
import cv2


def getScores(conf_matrix):
    if conf_matrix.sum() == 0:
        return 0, 0, 0, 0, 0, 0, 0
    with np.errstate(divide='ignore',invalid='ignore'):
        globalacc = np.diag(conf_matrix).sum() / float(conf_matrix.sum())
        classpre = np.diag(conf_matrix) / conf_matrix.sum(0).astype(float)
        classrecall = np.diag(conf_matrix) / conf_matrix.sum(1).astype(float)
        IU = np.diag(conf_matrix) / (conf_matrix.sum(1) + conf_matrix.sum(0) - np.diag(conf_matrix)).astype(float)
        pre = classpre[1]
        recall = classrecall[1]
        iou = IU[1]
        F_score = 2*(recall*pre)/(recall+pre)
        fpr = conf_matrix[0, 1] / float(conf_matrix[0, 0] + conf_matrix[0, 1])
        fnr = conf_matrix[1, 0] / float(conf_matrix[1, 0] + conf_matrix[1, 1])
    return globalacc, pre, recall, F_score, iou, fpr, fnr


def val_thresh(args, model, dataloader):
    print('start val!')
    # label_info = get_label_info(csv_path)
    conf_mat = np.zeros((256, 2, 2), dtype=np.float)

    pred_t_count = np.zeros(256, dtype=np.int64)
    pred_f_count = np.zeros(256, dtype=np.int64)
    with torch.no_grad():
        model.eval()

        for i, sample in enumerate(dataloader):
            image, depth, label = sample['image'], sample['depth'], sample['label']
            oriHeight, oriWidth = sample['oriHeight'], sample['oriWidth']
            oriWidth = oriWidth.cpu().numpy()[0]
            oriHeight = oriHeight.cpu().numpy()[0]
            
            if torch.cuda.is_available() and args.use_gpu:
                image = image.cuda()
                depth = depth.cuda()
                label = label.cuda()

            evidence, evidence_a, alpha, alpha_a = model(image, depth)
            s = torch.sum(alpha_a, dim=1, keepdim=True)
            b = alpha_a / (s.expand(alpha_a.shape))
            u = args.num_classes / s

            predict = b[:,1]
            predict = predict.view(args.crop_height, args.crop_width)
            predict = predict.detach().cpu().numpy()
            predict = np.array(predict)
            label = label.squeeze(0).cpu().numpy()
            label = cv2.resize(np.uint8(label), (oriWidth, oriHeight), interpolation=cv2.INTER_NEAREST)
            predict = cv2.resize(predict, (oriWidth, oriHeight))
            result = np.array(np.floor(255*(predict - predict.min()) / (predict.max()-predict.min())))
            result = result.astype(np.uint8)
            if args.dataset =='cityscapes':
                if np.sum(label > 0) < 100:
                    continue

            # # save result
            # save_dir = args.save_path
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # save_name = img.split('_')[0]+'_road_'+img.split('_')[1]
            # save_path = os.path.join(save_dir, save_name)
            # cv2.imwrite(save_path, np.uint8(result))

            gt_t = label == 1
            gt_f = label != 1
            pred_t = result * gt_t
            pred_f = result * gt_f
            pred_t_count += np.bincount(pred_t[gt_t].flatten(), minlength=256)
            pred_f_count += np.bincount(pred_f[gt_f].flatten(), minlength=256)

        for thresh in range(255):
            tp = np.cumsum(pred_t_count)[-1] - np.cumsum(pred_t_count)[thresh]
            fn = np.cumsum(pred_t_count)[thresh]
            tn = np.cumsum(pred_f_count)[thresh]
            fp = np.cumsum(pred_f_count)[-1] - np.cumsum(pred_f_count)[thresh]
            conf_mat[thresh, :, :] += np.array([[tn, fp], [fn, tp]])
        globalacc = np.zeros(256)
        pre = np.zeros(256)
        recall = np.zeros(256)
        F_score = np.zeros(256)
        iou = np.zeros(256)
        fpr = np.zeros(256)
        fnr = np.zeros(256)
        for i in range(256):
            globalacc[i], pre[i], recall[i], F_score[i], iou[i], fpr[i], fnr[i] = getScores(conf_mat[i, :, :])
        
        if args.dataset != 'r2d':
            # Select the best result (MaxF)
            index = np.argmax(F_score)
        else:
            # F-score
            index = 128
        
        print(index / 255)
        print('globalacc: %.4f' % globalacc[index])
        print('pre : %.4f' % pre[index])
        print('recall: %.4f' % recall[index])
        print('F_score: %.4f' % F_score[index])
        print('iou: %.4f' % iou[index])

        return globalacc[index], pre[index], recall[index], F_score[index], iou[index], index/255

def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', action='store_true', default=False, help='predict on image')
    parser.add_argument('--video', action='store_true', default=False, help='predict on video')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='The path to the pretrained weights of model')
    parser.add_argument('--context_path', type=str, default="resnet101", help='The context path model you are using.')
    parser.add_argument('--num_classes', type=int, default=2, help='num of object classes (with void)')
    parser.add_argument('--data', type=str, default=None, help='Path to image or video for prediction')
    parser.add_argument('--dataset', type=str, default="kitti", help='Dataset you are using.')
    ## Cityscapes
    # parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped/resized input image to network')
    # parser.add_argument('--crop_width', type=int, default=1024, help='Width of cropped/resized input image to network')
    ## ORFD
    # parser.add_argument('--crop_height', type=int, default=480, help='Height of cropped/resized input image to network')
    # parser.add_argument('--crop_width', type=int, default=640, help='Width of cropped/resized input image to network')
    ## R2D
    # parser.add_argument('--crop_height', type=int, default=704, help='Height of cropped/resized input image to network')
    # parser.add_argument('--crop_width', type=int, default=1280, help='Width of cropped/resized input image to network')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to user gpu for training')
    parser.add_argument('--save_path', type=str, default=None, required=True, help='Path to save predict image')


    args = parser.parse_args(params)

    if args.dataset == 'cityscapes':
        test_set = CityscapesAugmentationFusionSegmentation(args, root=args.data, split='val')
    elif args.dataset == 'r2d':
        test_set = R2DAugmentationFusionSegmentation(args, root=args.data, split='testing')
    elif args.dataset == 'orfd':
        test_set = OrfdAugmentationFusionSegmentation(args, root=args.data, split='testing')

    dataloader_test = DataLoader(
        test_set,
        batch_size=1,
        shuffle=True,
        num_workers=8
    )
    
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = USNet(args.num_classes, args.context_path)

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # load pretrained model if exists
    if args.checkpoint_path is not None:
        print('load model from %s ...' % args.checkpoint_path)
        model.module.load_state_dict(torch.load(args.checkpoint_path))
        print('Done!')

    val_thresh(args, model, dataloader_test)


if __name__ == '__main__':
    params = [
        '--dataset', 'orfd', # change it to the dataset you want to test
        '--data', '/data/ORFD', # change it to your dataset path
        '--num_classes', '2',
        '--cuda', '0',
        '--context_path', 'resnet18',
        '--checkpoint_path', 'weight/orfd/best_model.pth', # change it to your parameter path
        '--save_path', 'result/test' # change it to your custom path
    ]
    main(params)

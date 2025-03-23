from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse
from dataset.kitti_bev_sne_augmentation_fusion import KittiBevSneAugmentationFusionSegmentation
import os
from model.usnet import USNet
from bev.BirdsEyeView import BirdsEyeView
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import tqdm
import numpy as np
from utils import poly_lr_scheduler, poly_lr_scheduler_2_group
import random
import cv2


# loss function
def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)

def mse_loss(p, alpha, c, global_step, annealing_step=1):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    m = alpha / S
    label = F.one_hot(p, num_classes=c)
    A = torch.sum((label - m) ** 2, dim=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    C = annealing_coef * KL(alp, c)
    return (A + B) + C

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

# different thresh
def val_thresh(args, model, dataloader):
    print('start val!')
    # label_info = get_label_info(csv_path)
    conf_mat = np.zeros((256, 2, 2), dtype=float)

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

            predict = b[:,1]
            predict = predict.view(args.crop_height, args.crop_width)
            predict = predict.detach().cpu().numpy()
            predict = np.array(predict)
            label = label.squeeze(0).cpu().numpy()
            label = cv2.resize(np.uint8(label), (oriWidth, oriHeight), interpolation=cv2.INTER_NEAREST)
            predict = cv2.resize(predict, (oriWidth, oriHeight))
            result = np.floor(255*(predict - predict.min()) / (predict.max()-predict.min()))
            result = result.astype(np.uint8)
            if np.sum(label > 0) < 100:
                continue
            
            # BEV class
            bev = BirdsEyeView()
            calib_path = sample['calib_path'][0]
            # Update calibration for Birds Eye View
            bev.setup(calib_path)
            # Compute Birds Eye View
            result = np.uint8(bev.compute(result))
            label = bev.compute(label)

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
        
        # Select the best result (MaxF)
        index = np.argmax(F_score)
        print(index / 255)
        print('globalacc: %.4f' % globalacc[index])
        print('pre : %.4f' % pre[index])
        print('recall: %.4f' % recall[index])
        print('Max F_score: %.4f' % F_score[index])
        print('iou: %.4f' % iou[index])

        return globalacc[index], pre[index], recall[index], F_score[index], iou[index], index/255

def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))

    max_F_score = 0
    step = 0
    lambda_epochs = 50
    for epoch in range(args.current_epoch, args.num_epochs):
        lr = poly_lr_scheduler_2_group(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        for i, sample in enumerate(dataloader_train):
            image, depth, label = sample['image'], sample['depth'], sample['label']
            if torch.cuda.is_available() and args.use_gpu:
                image = image.cuda()
                depth = depth.cuda()
                label = label.cuda()
            
            if args.loss == 'crossentropy':
                # crossentropy loss
                evidence_sup, alpha_sup, evidence, evidence_a, alpha, alpha_a = model(image, depth)
                
                label = label.flatten()
                loss = 0
                for v_num in range(len(alpha_sup)):
                    loss += ce_loss(label, alpha_sup[v_num], args.num_classes, epoch, lambda_epochs)
                for v_num in range(len(alpha)):
                    loss += ce_loss(label, alpha[v_num], args.num_classes, epoch, lambda_epochs)
                loss += 2 * ce_loss(label, alpha_a, args.num_classes, epoch, lambda_epochs)
                loss = torch.mean(loss)

            elif args.loss == 'mseloss':
                # mseloss
                evidence_sup, alpha_sup, evidence, evidence_a, alpha, alpha_a = model(image, depth)
                
                label = label.flatten()
                loss = 0
                for v_num in range(len(alpha_sup)):
                    loss += mse_loss(label, alpha_sup[v_num], args.num_classes, epoch, lambda_epochs)
                for v_num in range(len(alpha)):
                    loss += mse_loss(label, alpha[v_num], args.num_classes, epoch, lambda_epochs)
                loss += 2 * mse_loss(label, alpha_a, args.num_classes, epoch, lambda_epochs)
                loss = torch.mean(loss)


            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
            
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        if epoch % args.checkpoint_step == 0 and epoch != 0:
            if not os.path.isdir(args.save_model_path):
                os.makedirs(args.save_model_path)
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_model_path, 'usnet_latest.pth'))

        if epoch % args.validation_step == 0:
            if not os.path.isdir(args.save_model_path):
                os.makedirs(args.save_model_path)
            globalacc, pre, recall, F_score, iou, thresh = val_thresh(args, model, dataloader_val)
            file = open(os.path.join(args.save_model_path, 'F_score_val.txt'), mode='a+')
            file.write('epoch = %d, F_score = %f, thresh = %f\n' % (epoch, F_score, thresh))
            file.close()
            if F_score > max_F_score:
                max_F_score = F_score
                torch.save(model.module.state_dict(),
                           os.path.join(args.save_model_path, 'usnet_best.pth'))
            writer.add_scalar('val/global_acc', globalacc, epoch)
            writer.add_scalar('val/pre', pre, epoch)
            writer.add_scalar('val/recall', recall, epoch)
            writer.add_scalar('val/F_score', F_score, epoch)
            writer.add_scalar('val/iou', iou, epoch)


def main(params):
    # set initialization seed
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--loss', type=str, default="crossentropy", help='loss function you are using.')
    parser.add_argument('--augmentation', type=str, default="fusion", help='augmentation you are using.')
    parser.add_argument('--use_sne', action='store_true', default=False, help='Whether to user sne')
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--crop_height', type=int, default=384, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=1248, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet18", help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--cityscapes_path', type=str, default='', help='path of cityscapes data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=32, help='num of object classes (with void)')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--current_epoch', type=int, default=0, help='train from current_epoch')

    args = parser.parse_args(params)

    train_set = KittiBevSneAugmentationFusionSegmentation(args, root=args.data, split='training_all')
    val_set = KittiBevSneAugmentationFusionSegmentation(args, root=args.data, split='validating')

    
    dataloader_train = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    dataloader_val = DataLoader(
        val_set,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = USNet(args.num_classes, args.context_path)

    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    
    encoder_params = list(map(id, model.module.backbone.parameters()))
    base_params = filter(lambda p: id(p) not in encoder_params, model.parameters())

    optimizer = torch.optim.AdamW([{'params': base_params},
                                    {'params':model.module.backbone.parameters(), 'lr': args.learning_rate*0.1}],
                                    lr=args.learning_rate, betas=(0.9,0.999), weight_decay=0.01)

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')
    
    # train
    train(args, model, optimizer, dataloader_train, dataloader_val)


if __name__ == '__main__':
    params = [
        '--loss', 'mseloss',
        '--use_sne',
        '--num_epochs', '200',
        '--learning_rate', '1e-3',
        '--data', '/data/KITTI/KITTI',
        '--cityscapes_path', '/data/Cityscapes/Road',
        '--num_workers', '8',
        '--num_classes', '2',
        '--cuda', '0',
        '--batch_size', '2',
        '--save_model_path', './weight/kitti',
        '--context_path', 'resnet18',
        '--checkpoint_step', '1',
        '--validation_step', '1',
        '--current_epoch', '0',
        # '--pretrained_model_path', '',
    ]
    main(params)

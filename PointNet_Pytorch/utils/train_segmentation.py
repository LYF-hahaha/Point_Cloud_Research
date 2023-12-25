from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset
from pointnet.model import PointNetDenseCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--nepoch', type=int, default=5, help='number of epochs to train for')
    parser.add_argument('--output_folder', type=str, default='seg_model', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str,
                        default='/home/alex/Dataset/PC_Lesson/Chapter_05/shapenetcore_partanno_segmentation_benchmark_v0',
                        help="dataset path")
    parser.add_argument('--class_choice', type=str, default='Airplane', help="class_choice")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

    opt = parser.parse_args()
    print(opt)

    opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    # 重复运行本文件时，后面的torch.rand会生成同样的数
    torch.manual_seed(opt.manualSeed)

    # 不用modelnet40训练，因为没有gt
    dataset = ShapeNetDataset(
        root=opt.dataset,  # 数据集路径
        classification=False,  # 用于分割，不是分类
        class_choice=[opt.class_choice])  # 只对某一类进行分割
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    test_dataset = ShapeNetDataset(
        root=opt.dataset,
        classification=False,
        class_choice=[opt.class_choice],  # 也是对预定的类别进行测试
        split='test',
        data_augmentation=False)
    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    print(len(dataset), len(test_dataset))
    num_classes = dataset.num_seg_classes
    print('classes', num_classes)
    try:
        os.makedirs(opt.output_folder)
    except OSError:
        pass

    blue = lambda x: '\033[94m' + x + '\033[0m'

    classifier = PointNetDenseCls(k=num_classes,  # 指定分割的类别数
                                  feature_transform=opt.feature_transform)  # 是否进行特征变换

    # 如果有模型的话
    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()

    num_batch = len(dataset) / opt.batchSize

    loss_rec = SummaryWriter("loss_logs")
    acc_rec = SummaryWriter("acc_logs")

    for epoch in range(opt.nepoch):
        for i, data in enumerate(tqdm(dataloader, desc=f"Training Epoch= {epoch+1}/{opt.nepoch}")):
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            classifier = classifier.train()
            pred, trans, trans_feat = classifier(points)
            # 这里的维度要考虑一下的
            pred = pred.view(-1, num_classes)
            target = target.view(-1, 1)[:, 0] - 1
            # print(pred.size(), target.size())
            # 损失函数不变
            loss = F.nll_loss(pred, target)
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            loss_rec.add_scalar("train loss", loss.item(), epoch*len(dataloader)+i)
            acc_rec.add_scalar("train acc", correct.item()/float(opt.batchSize * 2500), epoch * len(dataloader) + i)
            # print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item()/float(opt.batchSize * 2500)))

            if i % 10 == 0:
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                pred = pred.view(-1, num_classes)
                target = target.view(-1, 1)[:, 0] - 1
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                loss_rec.add_scalar("val loss", loss.item(), epoch * len(dataloader) + i)
                acc_rec.add_scalar("val acc", correct.item() / float(opt.batchSize * 2500),
                                   epoch * len(dataloader) + i)
                # print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize * 2500)))
        scheduler.step()
        torch.save(classifier.state_dict(), '%s/seg_model_%s_%d.pth' % (opt.output_folder, opt.class_choice, epoch))
    loss_rec.close()
    acc_rec.close()

    # benchmark mIOU
    # 用测试集计算mIOU
    shape_ious = []
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(2)[1]

        pred_np = pred_choice.cpu().data.numpy()
        target_np = target.cpu().data.numpy() - 1

        for shape_idx in range(target_np.shape[0]):
            parts = range(num_classes)  # np.unique(target_np[shape_idx])
            part_ious = []
            # 逐个类别计算（这里应该是编号）
            for part in parts:
                I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))
                if U == 0:
                    iou = 1 # If the union of groundtruth and prediction points is empty, then count part IoU as 1
                else:
                    iou = I / float(U)
                part_ious.append(iou)
            shape_ious.append(np.mean(part_ious))

    print("mIOU for class {}: {}".format(opt.class_choice, np.mean(shape_ious)))


if __name__ == "__main__":
    train()

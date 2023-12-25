from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--num_points', type=int, default=7500, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--epoch', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--output_folder', type=str, default='cls_model', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, default='/home/alex/Dataset/PC_Lesson/Chapter_05/modelnet_40', help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='modelnet40', help="dataset type shapenet or modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    opt = parser.parse_args()
    print(opt)

    # 定义蓝色的RGB值
    blue = lambda x: '\033[94m' + x + '\033[0m'

    opt.manualSeed = random.randint(1, 10000)  # fix seed (随机选了一个)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)  # 设定随机种子
    torch.manual_seed(opt.manualSeed)  # 重复运行本文件时，后面的torch.rand会生成同样的数

    # 选一个数据集载入
    if opt.dataset_type == 'shapenet':
        dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            npoints=opt.num_points)

        test_dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    elif opt.dataset_type == 'modelnet40':
        train_dataset = ModelNetDataset(
            root=opt.dataset,
            npoints=opt.num_points,
            split='train')

        test_dataset = ModelNetDataset(
            root=opt.dataset,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False)
    else:
        exit('wrong dataset type')

    # 设定载入数据集的组织方式
    traindataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batchSize,
        shuffle=True,
        num_workers=int(opt.workers))

    testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.workers))

    print(len(train_dataset), len(test_dataset))
    num_classes = len(train_dataset.classes)
    print('classes', num_classes)

    # 生成文件输出路径
    try:
        os.makedirs(opt.output_folder)
    except OSError:
        pass

    # 指定分类器
    classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

    # 已有训练好的模型的话
    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    # 优化器、学习率、放GPU上
    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()

    # 算Batch数
    num_batch = len(train_dataset) / opt.batchSize

    loss_rec = SummaryWriter("loss_logs")
    acc_rec = SummaryWriter("acc_logs")
    # 开始epoch的训练
    for epoch in range(opt.epoch):
        # 一轮epoch后更新学习率
        # for i, data in enumerate(tqdm((traindataloader, 0), desc=f" Training Epoch= {epoch+1}/{opt.epoch}")):
        for i, data in enumerate(tqdm(traindataloader, desc=f" Training Epoch= {epoch + 1}/{opt.epoch}")):
            points, target = data
            # 取所有行的第0列
            # target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            # 梯度清零
            optimizer.zero_grad()
            # 分类器切换为训练模式（有梯度回传）
            classifier = classifier.train()
            # 预测一把
            pred, trans, trans_feat = classifier(points)
            # 负指数损失
            # log_softmax(x) + nn.NLLLoss ==> nn.CrossEntropyLoss
            loss = F.nll_loss(pred, target)
            # 加入T-Net网络
            if opt.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            loss_rec.add_scalar(f"Train Loss",
                                      loss.item(),
                                      epoch*len(traindataloader)+i)
            acc_rec.add_scalar(f"Train Accuracy",
                                     correct.item() / float(opt.batchSize),
                                     epoch*len(traindataloader)+i)
            # train_loss_rec.add_scalar(f"Train Loss in Epoch:{epoch+1}", loss.item(), i)
            # train_acc_rec.add_scalar(f"Train Accuracy in Epoch:{epoch + 1}", correct.item() / float(opt.batchSize), i)
            # print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), correct.item() / float(opt.batchSize)))

            # 每训练10次val一次
            if i % 10 == 0:
                # 用当前的下一个来val
                j, data = next(enumerate(testdataloader, 0))
                points, target = data
                # target = target[:, 0]
                points = points.transpose(2, 1)
                points, target = points.cuda(), target.cuda()
                classifier = classifier.eval()
                pred, _, _ = classifier(points)
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                loss_rec.add_scalar(f"Val Loss",
                                          loss.item(),
                                          epoch * len(traindataloader) + i)
                acc_rec.add_scalar(f"Val Accuracy",
                                         correct.item() / float(opt.batchSize),
                                         epoch * len(traindataloader) + i)
                # val_loss_rec.add_scalar(f"Val Loss in Epoch:{epoch + 1}", loss.item(), i)
                # val_acc_rec.add_scalar(f"Val Accuracy in Epoch:{epoch + 1}", correct.item() / float(opt.batchSize), i)
                # print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))
        scheduler.step()
        # 每个epoch模型都保存
        torch.save(classifier.state_dict(), '%s/cls_model_epoch%d.pth' % (opt.output_folder, epoch))
    loss_rec.close()
    acc_rec.close()

    total_correct = 0
    total_testset = 0
    # 用测试集测试
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        # target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]
    print("final accuracy {}".format(total_correct / float(total_testset)))


if __name__ == "__main__":
    train()

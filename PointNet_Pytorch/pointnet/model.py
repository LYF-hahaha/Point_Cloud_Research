from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


# 最开始3维的变换
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        # MLP 3升维64 kernel_size=1 (stride-1 padding=0 by default)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        # MLP 64升维1024 kernel_size=1 (stride-1 padding=0 by default)
        # 之所以分两步，是因为原文中就是分了两步的
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        # 定义全连接层（这里的输出需要是对应的预测类别数）
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        # 定义激活函数
        self.relu = nn.ReLU()
        # BatchNorm
        # 一个完整的MLP F.relu(self.bn(self.conv(x)))
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # numpy里 size表示所有元素的个数和; shape表示数组各个维度的大小
        # pytorch里 两个都是一样 表示数组各维度的大小
        # 这里是指一个点算一个batch
        batchsize = x.size()[0]
        # 点云升维成n×1024
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # 列 max pooling
        # 之所以是2，因为shape=[batch_size,row,col]
        # keepdim 表示是否需要保持输出的维度与输入一样
        # keepdim=True表示输出和输入的维度一样
        # keepdim=False表示输出的维度被压缩了，也就是输出会比输入低一个维度
        # 直接torch.max操作后x还不是能用于计算的tensor，得x[0]后才是
        x = torch.max(x, 2, keepdim=True)[0]
        # view()的作用相当于numpy中的reshape，重新定义矩阵的形状
        x = x.view(-1, 1024)

        # 将全局特征用MLP变成对应的预测类别数
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # 将学得的变换参数与单位矩阵相加（方便之后做乘法）
        # 新建一个3×3的I矩阵，并将其转为float
        # 将ndarray转为tensor，再用pytorch的方式reshape一下(view)
        # 最后repeat:复制batch_size个
        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


# 64×64维的变换
class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        # T-Net(做特征主方向转换的可学习网络)
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        # 点的个数
        n_pts = x.size()[2]
        # T-Net
        trans = self.stn(x)
        # (3,2500) → (2500,3)
        x = x.transpose(2, 1)
        # 不同维度的矩阵乘法
        x = torch.bmm(x, trans)
        # (2500,3) → (3,2500)
        x = x.transpose(2, 1)

        # 3维点云 → 64维特征
        x = F.relu(self.bn1(self.conv1(x)))
        # 开是否继续T-Net
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        # 点云特征继续升维成n×1024
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        # 看是否要求网络返回全局特征
        if self.global_feat:
            return x, trans, trans_feat
        else:
            # 如果用于分割，就把全局特征和64维点特征融合在一起
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


# 分类网络，接收的是Global Feature
class PointNetCls(nn.Module):
    def __init__(self, k=2, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        # 提取升维的点云特征
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    # 如果用于分类
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        # k分类问题
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        # 返回k分类的概率值
        return F.log_softmax(x, dim=1), trans, trans_feat


# 分割网络，接收的是 Global+Local Feature
class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k  # 分类的类别数
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=False,  # 实例化对象的时候就说明了要融合特征，不要全局的
                                 feature_transform=feature_transform)
        # 1088=64+1024
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        # 从n×128到n×m，n代表点的个数，m代表这个点属于m个类别的概率
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        x = F.log_softmax(x.view(-1, self.k), dim=-1)
        # n_pts是点的数量
        # 最后输出的k维向量表达了n_pts中点i属于各个类别的概率
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss


if __name__ == '__main__':
    # 生成一个shape=[32，3，2500]的随机张量
    sim_data = Variable(torch.rand(32, 3, 2500))
    trans = STN3d()
    out = trans(sim_data)
    print('\nstn', out.size())
    print('loss', feature_transform_regularizer(out))

    sim_data_64d = Variable(torch.rand(32, 64, 2500))
    trans = STNkd(k=64)
    out = trans(sim_data_64d)
    print('\nstn64d', out.size())
    print('loss', feature_transform_regularizer(out))

    pointfeat = PointNetfeat(global_feat=True)
    out, _, _ = pointfeat(sim_data)
    print('\nglobal feat', out.size())

    pointfeat = PointNetfeat(global_feat=False)
    out, _, _ = pointfeat(sim_data)
    print('\npoint feat', out.size())

    cls = PointNetCls(k = 40)
    out, _, _ = cls(sim_data)
    print('\nclass', out.size())

    seg = PointNetDenseCls(k = 3)
    out, _, _ = seg(sim_data)
    print('\nseg', out.size())

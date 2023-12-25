from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm 
import json
from plyfile import PlyData, PlyElement


def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))
    
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))


def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'modelnet40_train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('_')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


class ShapeNetDataset(data.Dataset):
    # 划分训练集和测试集
    # 将原始点云和gt按类别对应好，并返回包含其绝对路径的list or dict
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}   # 记录类别名及其编号
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}
        
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # print(self.cat)
        # 只对某一类进行分割
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}  # 记录类别的什么用的？
        # 训练&测试划分文件（其实这个划分也有讲究的吧）
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        # from IPython import embed; embed()
        # 按逗号load的？
        filelist = json.load(open(splitfile, 'r'))
        # 先按类别填充字典的"键"
        # meta的键是"字母"
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            # 这里的category在文件中已经是编码了
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                # 记录"原始点云和对应的分割gt"文件的绝对路径，并append成list
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid+'.pts'),
                                        os.path.join(self.root, category, 'points_label', uuid+'.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                # 类别名 点云文件 分割gt
                self.datapath.append((item, fn[0], fn[1]))
        # zip: 将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        # dict:
        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    # 返回下采样后的点云以及cls or seg inidex
    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        # 点云文件
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        # seg_gt文件
        seg = np.loadtxt(fn[2]).astype(np.int64)
        # print(point_set.shape, seg.shape)
        # 点云文件的点的个数与seg_gt中点的个数对齐
        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]
        # center
        point_set = point_set - np.expand_dims(np.mean(point_set, axis = 0), 0)
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis = 1)), 0)
        # scale
        point_set = point_set / dist

        if self.data_augmentation:
            # 随机生成旋转角度
            theta = np.random.uniform(0, np.pi*2)
            # 生成旋转矩阵（2×2的）
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter
        # seg也下采样
        seg = seg[choice]
        # 转换成tensor
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        # 存储所有点云文件名
        self.PC_file_names = []
        # split成 训练集or测试集
        with open(os.path.join(root, 'modelnet40_{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.PC_file_names.append(line.strip())  # 删除字符串前后（左右两侧）的空格或特殊字符

        # 类别名与编号对应表
        self.cat = {}
        # 获取当前脚本文件的绝对路径，并去掉文件名，返回文件路径
        # 将上述路径与'../misc/modelnet_id.txt'连在一起
        # 并打开该文件
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])
        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.PC_file_names[index]
        class_name = fn.split('_')[0]
        # cls = self.cat[fn.split('_')[0]]

        # Original
        # with open(os.path.join(self.root, fn), 'rb') as f:
        #     plydata = PlyData.read(f)
        # pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T

        pts = []
        with open(os.path.join(self.root+'/txt', fn[:-5], fn+'.txt'), 'rb') as file:
            lines = file.readlines()
            for line in lines:
                xyz = line.decode().strip().split(',')[:3]
                a = [0, 0, 0]
                a[0] = float(xyz[0])
                a[1] = float(xyz[1])
                a[2] = float(xyz[2])
                pts.append(a)
            pts = np.array(pts)

        # 在len(pts)里随机选npoints个点
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        # 点云数量统一
        point_set = pts[choice, :]
        # x,y,z方向n_point个点求均值
        # dim=0方向扩展维度
        # 将原点设置在point_set的中心处
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        # 各点到原点的欧氏距离
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        # 归一化？
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array(self.cat[class_name]).astype(np.int64))
        return point_set, cls

    def __len__(self):
        return len(self.PC_file_names)


if __name__ == '__main__':
    # dataset = sys.argv[1]
    # datapath = sys.argv[2]
    dataset = 'modelnet'
    datapath = '/home/alex/Dataset/PC_Lesson/modelnet_40'

    if dataset == 'shapenet':
        d = ShapeNetDataset(root = datapath, class_choice = ['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(), seg.type())

        d = ShapeNetDataset(root = datapath, classification = True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(),cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d = ModelNetDataset(root=datapath)
        print(f"The overall point number is:{len(d)}")
        print(d[0])

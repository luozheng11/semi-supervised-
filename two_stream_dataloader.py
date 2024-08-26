import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import itertools
from torch.utils.data.sampler import Sampler
import cv2

class MYDataset(Dataset):
    #这里的base_dir是数据集的根目录，split是数据集的划分，num是数据集的大小，transform是数据集的预处理，需要自己调整
    def __init__(self, base_dir='D:/project/', split='train', num=None, transform=None):
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = []
        if split == 'train':
            with open(self._base_dir+'data/train.txt', 'r') as f:
                self.image_list = f.readlines()
        elif split == 'val':
            with open(self._base_dir+'data/val.txt', 'r') as f:
                # D:\project\v - set\val.txt
                self.image_list = f.readlines()
        #注意这个image_list是一个列表，每个元素是一个字符串，需要去掉换行符，然后再用split切分，取第一个元素，就是图像的名字，可能会存在空格，不确定可以打印出来看看
        self.image_list = [item.replace('\n','').split('.')[0] for item in self.image_list]
        if num is not None:
            self.image_list = self.image_list[:num]
        print("total {} samples".format(len(self.image_list)))

    def __len__(self):
             return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        print(idx)
        # print(self.image_list)
        # print(image_name)
        #这里需要根据存放的路径来读取图像和标签
        image = cv2.imread(self._base_dir+"/data/images/"+image_name+".jpg", 0)
        label = cv2.imread(self._base_dir+"/data/labels/"+image_name+'.png', 0)
        #调整图像大小,并转化为张量
        if label is None:
            print("Label is None!")
        else:
            label = cv2.resize(label, (256, 256))
        image = cv2.resize(image,(256,256))
        image = torch.from_numpy(image).unsqueeze(0).to(torch.float32) / 255.
        label = torch.from_numpy(label).unsqueeze(0).to(torch.float32) / 255.
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample

def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())

def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    dataset中的每个数据都有对应的索引，这里将有标签数据的索引和无标签数据的索引分开，构建了一个双流采样器，分别从两个索引列表中采样
    这是对两个不同的数据列表同时迭代组成单个batch用的。相当于是控制了dataloader的采样方式。
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

# dataset = MYDataset(split='train')
# print(dataset[0])

# a = list(range(0,10))
# b = list(range(10,50))
#
# batch_sampler = TwoStreamBatchSampler(a, b, 4, 3)
#
# print(len(batch_sampler))
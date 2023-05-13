import os
from torch.utils.data.dataset import Dataset
import torch
import numpy as np
from skimage import io
from PIL import Image
import torchvision
from torch.autograd import Variable


# 自定义数据集加载类
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 是否对图片进行变换
        self.images = os.listdir(self.root_dir)  # 获取图片文件名

    def __len__(self):
        return len(self.images)  # 返回图片的长度

    def __getitem__(self, item):
        # item = int(item.item())
        image_index = self.images[item]  # 根据索引获取图片
        image_path = os.path.join(self.root_dir, image_index)  # 获取索引图片的路径
        image_io = Image.open(image_path)
        # image_io = image_io.convert('L')
        img_label = image_path.split('/')[-1].split('.')[0][0]
        label = ord(img_label) - ord('A')
        label = np.array(label)

        if self.transform:
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(224), torchvision.transforms.ToTensor()])
            image_io = transform(image_io)
            label = Variable(torch.from_numpy(label)).type(torch.LongTensor)
        return image_io, label


class CustomDataset1(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 是否对图片进行变换
        self.images = os.listdir(self.root_dir)  # 获取图片文件名

    def __len__(self):
        return len(self.images)  # 返回图片的长度

    def __getitem__(self, item):
        # item = int(item.item())
        image_index = self.images[item]  # 根据索引获取图片
        image_path = os.path.join(self.root_dir, image_index)  # 获取索引图片的路径
        image_io = Image.open(image_path)
        # image_io = image_io.convert('L')
        img_label = image_path.split('/')[-1].split('.')[0][0]

        if img_label == 'A' or img_label == 'B' or img_label == 'C' or img_label == 'D':
            label = ord('A') - ord('A')
        else:
            label = ord('B') - ord('A')

        label = np.array(label)

        if self.transform:
            transform = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(224), torchvision.transforms.ToTensor()])
            image_io = transform(image_io)
            label = Variable(torch.from_numpy(label)).type(torch.LongTensor)
        return image_io, label


class SamplerDef(object):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


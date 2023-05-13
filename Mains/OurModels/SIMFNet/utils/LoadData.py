import os
from PIL import Image
import torchvision
from torch.autograd import Variable
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import csv


class SamplerDef(object):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

# root_dir ： domain/tr..td..cvd/A..B..C..D..E..F..G
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None, view_type=None, data_type=None):
        super(CustomDataset, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.view_type = view_type
        self.data_type = data_type

        train_set, val_set, test_set = self.split_data()
        if self.data_type == 'train':
            self.images = train_set
        elif self.data_type == 'val':
            self.images = val_set
        elif self.data_type == 'test':
            self.images = test_set

    def __getitem__(self, item):
        image_path = self.images[item]
        image_io = Image.open(image_path)
        image_label = image_path.split('/')[-2]

        label = ord(image_label) - ord('A')
        label = np.array(label)
        if self.transform:
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
            ])
            image_io = transform(image_io)
            label = Variable(torch.from_numpy(label)).type(torch.LongTensor)
        return image_io, label, image_path

    def __len__(self):
        return len(self.images)

    def split_data(self):
        train_set, val_set, test_set = [], [], []

        for i in range(len(self.root_dir)):
            root_dir = self.root_dir[i]
            dir_path = os.path.join(root_dir, self.view_type)
            for path_name in os.listdir(dir_path):
                img_data_path = os.path.join(dir_path, path_name)
                img_data = os.listdir(img_data_path)
                img_data.sort()

                train_data = img_data[: int(len(img_data) * 0.64)]
                val_data = img_data[int(len(img_data) * 0.64): int(len(img_data) * 0.8)]
                test_data = img_data[int(len(img_data) * 0.8): int(len(img_data))]

                for train_image in train_data:
                    train_image_path = os.path.join(img_data_path, train_image)
                    train_set.append(train_image_path)
                for val_image in val_data:
                    val_image_path = os.path.join(img_data_path, val_image)
                    val_set.append(val_image_path)
                for test_image in test_data:
                    test_image_path = os.path.join(img_data_path, test_image)
                    test_set.append(test_image_path)
        return train_set, val_set, test_set

# root_dir ： domain/tr..td..cvd/A..B..C..D..E..F..G
class CustomDataset1(Dataset):
    def __init__(self, root_dir, transform=None, view_type=None, data_type=None):
        super(CustomDataset1, self).__init__()
        self.root_dir = root_dir  # 根路径
        self.transform = transform  # 是否变换
        self.view_type = view_type
        self.data_type = data_type

        train_set, val_set, test_set = self.split_data()
        if self.data_type == 'train':
            self.images = train_set
        elif self.data_type == 'val':
            self.images = val_set
        elif self.data_type == 'test':
            self.images = test_set

    def __getitem__(self, item):
        image_path = self.images[item]  # 根据索引获取图片
        image_io = Image.open(image_path)  # 获取PIL图片
        image_label = image_path.split('/')[-2]  # 获取label的ASCII
        label = ord(image_label) - ord('A')  # 标签ASCII转数字
        label = np.array(label)
        if self.transform:  # 是否进行变换
            transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
            ])
            image_io = transform(image_io)
            label = Variable(torch.from_numpy(label)).type(torch.LongTensor)
        return image_io, label, image_path

    def __len__(self):
        return len(self.images)

    def split_data(self):
        train_set, val_set, test_set = [], [], []

        for i in range(len(self.root_dir)):
            root_dir = self.root_dir[i]
            dir_path = os.path.join(root_dir, self.view_type)  # domain/td/
            for path_name in os.listdir(dir_path):  # domain/td/a/
                img_data_path = os.path.join(dir_path, path_name)
                img_data = os.listdir(img_data_path)
                img_data.sort()

                train_data = img_data[: int(len(img_data) * 0.)]
                val_data = img_data[int(len(img_data) * 0.): int(len(img_data) * 0.)]
                test_data = img_data[int(len(img_data) * 0.): int(len(img_data))]

                for train_image in train_data:
                    train_image_path = os.path.join(img_data_path, train_image)
                    train_set.append(train_image_path)
                for val_image in val_data:
                    val_image_path = os.path.join(img_data_path, val_image)
                    val_set.append(val_image_path)
                for test_image in test_data:
                    test_image_path = os.path.join(img_data_path, test_image)
                    test_set.append(test_image_path)
        return train_set, val_set, test_set
        
def get_dataloader(dataset, batch_size, joint_idx=None):
    if joint_idx == None:
        dataloaders = DataLoader(dataset=dataset, batch_size=batch_size,
                                      shuffle=True, drop_last=True)
    else:
        tr_dataloder = DataLoader(dataset=dataset['tr'], batch_size=batch_size,
                                  sampler=joint_idx, shuffle=False, drop_last=True)
        td_dataloder = DataLoader(dataset=dataset['td'], batch_size=batch_size,
                                  sampler=joint_idx, shuffle=False, drop_last=True)
        cvd_dataloder = DataLoader(dataset=dataset['cvd'], batch_size=batch_size,
                                   sampler=joint_idx, shuffle=False, drop_last=True)
        dataloaders = {'tr': tr_dataloder, 'td': td_dataloder, 'cvd': cvd_dataloder}

    return dataloaders

import sys
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
import warnings
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))


label_list = ['distress', 'drown', 'freestyle', 'breaststroke', 'backstroke', 'pull with a ring',
                  'swim with a ring', ' float with a ring', 'wave', 'frolic']


def write_data_to_csv(path, hist, data_type=None, epoch=None):
    if data_type == 'train':
        path = path + "train.csv"
        file = open(path, 'a+', encoding='utf-8', newline='')
        csv_writer = csv.writer(file)
        csv_writer.writerow(
            [epoch, hist['train_acc'][-1].cpu().numpy(), hist['train_loss'][-1]])

    elif data_type == 'val':
        path = path + "val.csv"
        file = open(path, 'a+', encoding='utf-8', newline='')
        csv_writer = csv.writer(file)
        csv_writer.writerow([epoch, hist['val_acc'][-1].cpu().numpy(), hist['val_loss'][-1]])

    elif data_type == 'test':
        if epoch == 'train':
            path = path + "train_predict.csv"
        elif epoch == 'test':
            path = path + "test_predict.csv"

        file = open(path, 'a+', encoding='utf-8', newline='')
        csv_writer = csv.writer(file)
        for i in range(len(hist['image_path'])):
            csv_writer.writerow([hist['image_path'][i], label_list[hist['real_label'][i]],
                                 label_list[hist['predict_label'][i][0]], hist['logit_vector'][i]])

    else:
        warnings.warn("Please check out your data type!")

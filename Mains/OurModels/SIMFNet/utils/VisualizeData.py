import sys
import os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

import torch
import time
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch import optim
import numpy as np

from torch.autograd import Variable
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from sup_models import DCNN_MFS, ADGCN, GMFN, MDFRadarNet, TSFNet, LabelSmoothing
from mpl_toolkits.mplot3d import Axes3D


from utils import LoadData, Params
import itertools


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def confusion_matrix(preds, labels, num_classes=10):
    conf_matrix = torch.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            conf_matrix[j][i] = ((preds == i) & (labels == j)).sum().item()
    return conf_matrix


def plot_confusion_matrix(path, cm, classes, normalize=True, cmap=plt.cm.jet):

    plt.rc('font', family='Times New Roman')
    plt.rcParams['figure.dpi'] = 396  
    fig = plt.figure(figsize=(8,8))  
    path = path + str("confusion_matrix.png")

    if normalize:
        cm = cm / cm.sum(axis=1)[:, np.newaxis] * 100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm = cm.T
    plt.imshow(cm.T, interpolation='nearest', cmap=cmap)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")
    ax = plt.gca()  # 获得当前axis
    left, right = plt.xlim()  # 获得x轴最大最小值
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    print(cm.shape[0])
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(i, j, num,
                 va='center',
                 ha="center",
                 color="white" if i == j else "black",
                 fontsize=10)

    #plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    
    
def supervised_cmap(test_dataset, model_path):
    png_path = model_path
    model_path += "model.pth"
    indices = torch.randperm(len(test_dataset['tr']))
    mySampler = LoadData.SamplerDef(indices=indices)
    test_dataloader = LoadData.get_dataloader(test_dataset, batch_size=Params.vis_batch_size,
                                              joint_idx=mySampler)

    model = torch.load(model_path)
    model.eval()

    tr_iter, td_iter, cvd_iter = iter(test_dataloader['tr']), iter(test_dataloader['td']), iter(test_dataloader['cvd'])

    with torch.no_grad():
        tr_x, tr_y, _ = next(tr_iter)
        td_x, _, _ = next(td_iter)
        cvd_x, _, _ = next(cvd_iter)
        s_x = torch.cat((tr_x, td_x, cvd_x), dim=0)

        if Params.use_gpu:
            s_x, s_y = s_x.to(device), tr_y.to(device)
        else:
            s_x, s_y = s_x, tr_y

        output = model.predict(s_x)
        prediction = torch.max(output,1)[1]
        prediction = prediction.cpu().detach().numpy()
        s_y = s_y.cpu().detach().numpy()
        
        cmatrix = confusion_matrix(prediction, labels=s_y, num_classes=10)
        
        types = np.array(['A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6',
                          'A7', 'A8', 'A9'])
        plot_confusion_matrix(png_path, cmatrix, classes=types, normalize=True, cmap=plt.cm.jet)
        
        
        
def supervised_tsne(train_dataset, test_dataset, model_path):
    png_path = model_path
    model_path += 'model.pth'
    
    train_indices = torch.randperm(len(train_dataset['tr']))
    train_Sampler = LoadData.SamplerDef(indices=train_indices)
    train_dataloader = LoadData.get_dataloader(train_dataset, batch_size=Params.vis_batch_size, joint_idx=train_Sampler)
    
    test_indices = torch.randperm(len(test_dataset['tr']))
    test_Sampler = LoadData.SamplerDef(indices=test_indices)
    test_dataloader = LoadData.get_dataloader(test_dataset, batch_size=Params.vis_batch_size, joint_idx=test_Sampler)

    model = torch.load(model_path)
    model.eval()
    
    train_tr_iter, train_td_iter, train_cvd_iter = iter(train_dataloader['tr']), iter(train_dataloader['td']), iter(train_dataloader['cvd'])
    test_tr_iter, test_td_iter, test_cvd_iter = iter(test_dataloader['tr']), iter(test_dataloader['td']), iter(test_dataloader['cvd'])

    with torch.no_grad():
        train_tr_x, train_tr_y, _ = next(train_tr_iter)
        train_td_x, _, _ = next(train_td_iter)
        train_cvd_x, _, _ = next(train_cvd_iter)
        train_x = torch.cat((train_tr_x, train_td_x, train_cvd_x), dim=0)
        
        test_tr_x, test_tr_y, _ = next(test_tr_iter)
        test_td_x, _, _ = next(test_td_iter)
        test_cvd_x, _, _ = next(test_cvd_iter)
        test_x = torch.cat((test_tr_x, test_td_x, test_cvd_x), dim=0)
        
        if Params.use_gpu:
            train_x = Variable(train_x.to(device))
            test_x = Variable(test_x.to(device))
        else:
            train_x = Variable(train_x)
            test_x = Variable(test_x)

        train_features = model.predict(train_x)
        train_features = train_features.cpu().data.numpy()
        
        test_features = model.predict(test_x)
        test_features = test_features.cpu().data.numpy()
        

        train_features = TSNE(n_components=2, learning_rate="auto", init='pca', random_state=0).fit_transform(
            train_features)
        test_features = TSNE(n_components=2, learning_rate="auto", init='pca', random_state=0).fit_transform(
            test_features)
            
        plt.scatter(train_features[:, 0], train_features[:, 1], color='r', label='Source Domain')
        plt.scatter(test_features[:, 0], test_features[:, 1], color='b', label='Target Domain')
        
        path = png_path + str("feature_map.png")
        plt.legend(bbox_to_anchor=(0.65, 0.3), loc=2, prop={'size': 10})
        plt.savefig(path, bbox_inches='tight')
        plt.show()
        
def supervised_tsne1(test_dataset, model_path):
    png_path = model_path
    model_path += 'model.pth'
    
    
    test_indices = torch.randperm(len(test_dataset['tr']))
    test_Sampler = LoadData.SamplerDef(indices=test_indices)
    test_dataloader = LoadData.get_dataloader(test_dataset, batch_size=Params.vis_batch_size, joint_idx=test_Sampler)

    model = torch.load(model_path)
    model.eval()
    
    test_tr_iter, test_td_iter, test_cvd_iter = iter(test_dataloader['tr']), iter(test_dataloader['td']), iter(test_dataloader['cvd'])
    fig = plt.figure(figsize=(5,5))  
    
    with torch.no_grad():
        
        test_tr_x, test_tr_y, _ = next(test_tr_iter)
        test_td_x, _, _ = next(test_td_iter)
        test_cvd_x, _, _ = next(test_cvd_iter)
        test_x = torch.cat((test_tr_x, test_td_x, test_cvd_x), dim=0)
        
        if Params.use_gpu:
            test_x = Variable(test_x.to(device))
        else:
            test_x = Variable(test_x)
        
        test_features = model.predict(test_x)
        test_features = test_features.cpu().data.numpy()
        
        test_features = TSNE(n_components=2, learning_rate="auto", init='pca', random_state=0).fit_transform(
            test_features)
            
        color = ['#DC143C', '#FF4500', '#00FF00', '#0000FF', '#4B0082', '#FFFF00', '#9400D3', '#4169E1', '#00FFFF', '#2F4F4F']
        marker = ['o', 'v', '<', '>', 'p', 'x', '+', 'd', '1', '^']    
        str_labels = ['distress', 'drown', 'freestyle', 'breaststroke', 'backstroke', 'pull a ring', 'swim with a ring', 'float with a ring', ' wave', 'frolic']
        for i in range(test_features.shape[0]):
            plt.scatter(test_features[i, 0], test_features[i, 1], color=color[test_tr_y[i]], label=str_labels[test_tr_y[i]], marker=marker[test_tr_y[i]])
        
        path = png_path + str("feature_map.png")
        # plt.legend(bbox_to_anchor=(1.0, 0.5), loc=2, prop={'size': 10})
        plt.savefig(path, bbox_inches='tight')
        plt.show()
        
               
    

def supervised_tsne2(test_dataset, model_path):
    png_path = model_path
    model_path += 'model.pth'
    indices = torch.randperm(len(test_dataset['tr']))
    mySampler = LoadData.SamplerDef(indices=indices)
    test_dataloader = LoadData.get_dataloader(test_dataset, batch_size=Params.vis_batch_size,
                                              joint_idx=mySampler)

    model = torch.load(model_path)
    model.eval()

    tr_iter, td_iter, cvd_iter = iter(test_dataloader['tr']), iter(test_dataloader['td']), iter(test_dataloader['cvd'])

    with torch.no_grad():
        tr_x, tr_y, _ = next(tr_iter)
        td_x, _, _ = next(td_iter)
        cvd_x, _, _ = next(cvd_iter)
        s_x = torch.cat((tr_x, td_x, cvd_x), dim=0)

        if Params.use_gpu:
            s_x = Variable(s_x.to(device))
        else:
            s_x = Variable(s_x)

        features = model.predict(s_x)
        features = features.cpu().data.numpy()

        features = TSNE(n_components=3, learning_rate="auto", init='pca', random_state=0).fit_transform(
            features)

        fig = plot_embedding(features, tr_y, png_path)


def plot_embedding(data, label, path):
    
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)  
    plt.rc('font', family='Times New Roman')
   
    plt.rcParams['figure.dpi'] = 396  
    fig = plt.figure()  # figsize=(5, 5)
    ax = Axes3D(fig)

    point = []
    color = ['#DC143C', '#FF4500', '#00FF00', '#0000FF', '#4B0082', '#FFFF00', '#9400D3', '#4169E1', '#00FFFF', '#2F4F4F']
    marker = ['o', 'v', '<', '>', 'p', 'x', '+', 'd', '1', 'o']
    
    for i in range(data.shape[0]):
        ax.text(data[i, 0], data[i, 1], data[i, 2],str(label[i]), color=color[label[i]])
    plt.show()
    plt.xticks()  
    plt.yticks()
   
    #ax.set_xlabel('dimension 1')
    #ax.set_ylabel('dimension 2')
    #ax.set_zlabel('dimension 3')

    plt.tick_params(labelsize=10)  
    ax = plt.gca()  
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['right'].set_linewidth(1.2)
    ax.spines['top'].set_linewidth(1.2)

    path = path + str("feature_map.png")
    plt.savefig(path, bbox_inches='tight')
    # plt.legend()
    
    return fig
    
    
    
    
def inference_time_compute(model):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    random_input = torch.randn(3, 3, 224, 224).to(device)
    epochs = 300
    times = torch.zeros(epochs)
    model.eval()
    for _ in range(10):
        _ = model(random_input)

    with torch.no_grad():
        for epoch in range(epochs):
            starter.record()
            _ = model(random_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            times[epoch] = curr_time
    mean_time = times.mean().item()
    print("Inference Time: {:.4f}, FPS: {}".format(mean_time, 1000/mean_time))


def inference_time_test():
  model = GMFN.GMFN().to(device)
  inference_time_compute(model)
  
  model = DCNN_MFS.DCNN_MFSRM().to(device)
  inference_time_compute(model)
  
  model = MDFRadarNet.MDFRadarNet().to(device)
  inference_time_compute(model)
  
  model = TSFNet.TSFNet().to(device)
  inference_time_compute(model)
  
  model = ADGCN.ADGCN().to(device)
  inference_time_compute(model)
  
import torch
from utils import Params
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))


def train(dataloader, model, device, cls_criterion, optimizer, lr_scheduler, hist):
    model.train()

    tr_iter, td_iter, cvd_iter = iter(dataloader['tr']), iter(dataloader['td']), iter(dataloader['cvd'])
    avg_cls_loss = 0
    avg_acc = 0

    for batch_idx in range(len(dataloader['tr'])):
        tr_x, tr_y, _ = next(tr_iter)
        td_x, td_y, _ = next(td_iter)
        cvd_x, cvd_y, _ = next(cvd_iter)

        s_x = torch.cat((tr_x, td_x, cvd_x), dim=0)
        s_y = F.one_hot(tr_y, 10)

        if Params.use_gpu:
            s_x, s_y, tr_y = Variable(s_x.to(device)), Variable(s_y.to(device)), Variable(tr_y.to(device))
        else:
            s_x, s_y, tr_y = Variable(s_x), Variable(s_y), Variable(tr_y)

        optimizer.zero_grad()

        cls_vector, ae_loss = model(s_x)

        class_loss = cls_criterion(cls_vector, s_y)
        loss = class_loss + ae_loss

        avg_cls_loss += loss.item()

        pred = cls_vector.data.max(1, keepdim=True)[1]
        avg_acc += pred.eq(tr_y.data.view_as(pred)).cpu().sum()

        class_loss.backward()
        optimizer.step()
        lr_scheduler.step()

    avg_acc = 100. * avg_acc / len(dataloader['tr'].dataset)
    avg_cls_loss = avg_cls_loss / len(dataloader['tr'])
    hist['train_acc'].append(avg_acc)
    hist['train_loss'].append(avg_cls_loss)
    print('Train Accuracy: {:.4f}%, Train Loss: {:.4f}'.format(avg_acc, avg_cls_loss))


def validate(dataloader, model, device, cls_criterion, hist):
    model.eval()

    tr_iter, td_iter, cvd_iter = iter(dataloader['tr']), iter(dataloader['td']), iter(dataloader['cvd'])
    loss = 0
    acc = 0
    with torch.no_grad():
        for batch_idx in range(len(dataloader['tr'])):
            tr_x, tr_y, _ = next(tr_iter)
            td_x, _, _ = next(td_iter)
            cvd_x, _, _ = next(cvd_iter)

            s_x = torch.cat((tr_x, td_x, cvd_x), dim=0)
            s_y = F.one_hot(tr_y, 10)

            if Params.use_gpu:
                s_x, s_y, tr_y = Variable(s_x.to(device)), Variable(s_y.to(device)), Variable(tr_y.to(device))
            else:
                s_x, s_y, tr_y = Variable(s_x), Variable(s_y), Variable(tr_y)

            cls_out = model.predict(s_x)
            cls_loss = cls_criterion(cls_out, s_y)
            loss += cls_loss.item()

            pred1 = cls_out.max(1, keepdim=True)[1]
            acc += pred1.eq(tr_y.data.view_as(pred1)).cpu().sum()

        val_correct = 100. * acc / len(dataloader['tr'].dataset)
        val_loss = loss / len(dataloader['tr'])
        print('Val Accuracy: {:.4f}%, Val Loss: {:.4f}'.format(val_correct, val_loss))
        hist['val_acc'].append(val_correct)
        hist['val_loss'].append(val_loss)


def predict(dataloader, model, device, pred_hist):
    model.eval()
    tr_iter, td_iter, cvd_iter = iter(dataloader['tr']), iter(dataloader['td']), \
                                 iter(dataloader['cvd'])

    acc = 0
    with torch.no_grad():
        for batch_idx in enumerate(dataloader['tr']):
            tr_x, tr_y, tr_path = next(tr_iter)
            td_x, _, _ = next(td_iter)
            cvd_x, _, _ = next(cvd_iter)
            s_x = torch.cat((tr_x, td_x, cvd_x), dim=0)
            # s_y = F.one_hot(tr_y, 10)
            s_y = tr_y

            if Params.use_gpu:
                s_x, s_y, tr_y = Variable(s_x.to(device)), Variable(s_y.to(device)), Variable(tr_y.to(device))
            else:
                s_x, s_y, tr_y = Variable(s_x), Variable(s_y), Variable(tr_y)

            cls_out = model.predict(s_x)
            pred_label = np.argmax(cls_out.cpu().data.numpy(), axis=1)
            acc += 1 if pred_label == s_y.data.item() else 0

            pred_hist['image_path'].append(tr_path)
            pred_hist['real_label'].append(s_y.data.item())
            pred_hist['predict_label'].append(pred_label)
            pred_hist['logit_vector'].append(cls_out.cpu().data.numpy()[0])
        acc = acc / len(dataloader['tr'].dataset) * 100.
        print('Test Accuracy: {:4f}'.format(acc))

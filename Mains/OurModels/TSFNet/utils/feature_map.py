import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image

model = torch.load('./result/fusion_net/best_model_1.pth')

img1 = cv2.imread('./Data_v3/train_rt/BRT_21.jpg')
img1 = Image.fromarray(img1)

img2 = cv2.imread('./Data_v3/train_md/BMD_21.jpg')
img2 = Image.fromarray(img2)

img3 = cv2.imread('./Data_v3/train_cvd/BCVD_21.jpg')
img3 = Image.fromarray(img3)

img1 = transforms.Resize(224)(img1)
img_tensor1 = transforms.ToTensor()(img1).cuda()
plt.imshow(img_tensor1.transpose(2, 0).transpose(1, 0).cpu())
plt.show()
img_tensor1 = img_tensor1.unsqueeze(0)

img2 = transforms.Resize(224)(img2)
img_tensor2 = transforms.ToTensor()(img2).cuda()
print("1.", np.shape(img_tensor2))
plt.imshow(img_tensor2.transpose(2, 0).transpose(1, 0).cpu())
print("2.", np.shape(img_tensor2))
plt.show()
img_tensor2 = img_tensor2.unsqueeze(0)
print("3.", np.shape(img_tensor2))

img3 = transforms.Resize(224)(img3)
img_tensor3 = transforms.ToTensor()(img3).cuda()
plt.imshow(img_tensor3.transpose(2, 0).transpose(1, 0).cpu())
plt.show()
img_tensor3 = img_tensor3.unsqueeze(0)


model.eval()
with torch.no_grad():
    y1 = model.encoder1(img_tensor1)
    d1 = model.decoder1(y1)
    print(np.shape(d1))

    d1 = d1.squeeze(0).cpu().detach().numpy()

    d1 = d1.swapaxes(0, 1)
    d1 = d1.swapaxes(1, 2)

    print(np.shape(d1))
    plt.imshow(d1)
    plt.savefig("./maps/encoder_{}.jpg".format(1))
    plt.show()

'''
    y1 = y1.transpose(3, 1)
    y1 = y1.squeeze(2).squeeze(0).cpu().detach().numpy()  # 56 * 64
'''
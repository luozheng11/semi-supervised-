import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from network import UNet_Dual  # 假设这是模型定义的文件
from two_stream_dataloader import MYDataset  # 假设这是数据集定义的文件
from metrics import test_single_volume_dual  # 假设这是评估函数的文件
from torchvision.utils import save_image  # 用于保存图像

from torchvision.utils import save_image
import numpy as np
import torch

def save_images(image, filename):
    """
    保存图像到指定路径。

    参数:
    image (Tensor 或 numpy.ndarray): 要保存的图像数据。
    filename (str): 保存图像的文件路径。

    返回:
    None.
    """
    # 如果传入的是numpy数组，先转换为张量
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()

    # 确保张量在正确的设备上
    image = image.to(torch.device('cpu'))

    # 保存图像
    save_image(image, filename)

model_path = 'D:/project/best-model/70%_5600_dice_0.8384.pth'
num_classes = 2
img_size = 256
img_folder_path = r'D:/project/test'
save_dir = r'D:/project/output'
threshold = 0.5
# 加载模型参数
model = UNet_Dual(in_chns=1, class_num=num_classes)
model.load_state_dict(torch.load(model_path))
model.eval()  # 设置为评估模式

for img_name in os.listdir(img_folder_path):
    if img_name.endswith('.jpg'):
        img_path = os.path.join(img_folder_path,img_name)
        img = cv2.imread(img_path,0)
        img = cv2.resize(img,(img_size,img_size))
        input_img = torch.from_numpy(img).unsqueeze(0).to(torch.float32)/255.

        with torch.no_grad():
            outputs = model(input_img.view(1,1,img_size,img_size))
        _, preds = outputs

        preds = torch.argmax(torch.softmax(preds, dim=1), dim=1)
        preds = preds.detach().cpu().numpy().squeeze()  # 转换为numpy数组并去掉多余维度
        preds[preds>0.5]  = 255
        preds[preds<=0.5] = 0

        pred_image = save_images(preds, os.path.join(save_dir, img_name.replace('jpg','png')))

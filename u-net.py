
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import Dataset
import os

class DoubleConv(nn.Module):
    """两次卷积和批量归一化层的块"""

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()
        self.enc1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = DoubleConv(256, 512)

        # 瓶颈层
        self.bottleneck = DoubleConv(512, 1024)

        # 解码器
        self.dec1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(128, 64)

        # 最终输出层
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器
        enc1 = self.enc1(x)
        enc2 = self.pool1(self.enc2(enc1))
        enc3 = self.pool2(self.enc3(enc2))
        enc4 = self.pool3(self.enc4(enc3))

        # 瓶颈层
        bottleneck = self.bottleneck(enc4)

        # 解码器和跳跃连接
        dec1 = self.dec1(bottleneck)
        up2 = self.up2(dec1)
        dec2 = self.dec2(up2 + enc3)
        up3 = self.up3(dec2)
        dec3 = self.dec3(up3 + enc2)
        up4 = self.up4(dec3)
        dec4 = self.dec4(up4 + enc1)

        # 最终输出
        final_out = self.final_conv(dec4)
        return final_out


# 实例化模型
model = UNet(in_channels=3, out_channels=1)  # 假设输入图像为3通道，输出为1通道的二值分割图


### 训练U-Net模型
# 定义数据集
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        # 加载图像和掩码文件路径
        self.image_paths = sorted([os.path.join(image_dir, x) for x in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, x) for x in os.listdir(mask_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像和掩码
        image = Image.open(self.image_paths[idx])
        mask = Image.open(self.mask_paths[idx])

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# 数据预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

# 实例化数据集和数据加载器
image_dir = 'path_to_images'  # 替换为图像文件夹路径
mask_dir = 'path_to_masks'  # 替换为掩码文件夹路径
dataset = CustomDataset(image_dir, mask_dir, transform=transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=1e-4)

# 训练模型
model.train()
for epoch in range(num_epochs):
    for images, masks in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

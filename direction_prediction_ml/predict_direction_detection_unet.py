import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, random_split, DataLoader
import lightning as L
import pandas as pd
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch import Trainer
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
from datetime import timedelta
import random

class FishImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = read_image(img_path).float() / 255.0
        if self.transform:
            image = self.transform(image)
        flip = int(row['flip'])
        return image, torch.tensor(flip)

class GroupedFishDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.grouped = list(self.df.groupby('fish_id'))

    def __len__(self):
        return len(self.grouped)

    def __getitem__(self, idx):
        fish_id, group_df = self.grouped[idx]
        images = []
        flips = []

        for _, row in group_df.iterrows():
            img_path = os.path.join(self.img_dir, row['filename'])
            image = read_image(img_path).float() / 255.0
            if self.transform:
                image = self.transform(image)
            images.append(image)
            flips.append(int(row['flip']))  # Ensure int

        images = torch.stack(images)         # shape: [N, C, H, W]
        flips = torch.tensor(flips)          # shape: [N]
        
        return images, flips

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class DirectionDetectionUNet(L.LightningModule):
    def __init__(self, n_channels=1, feature_scale=4):
        super().__init__()
        self.unet = UNet(n_channels=n_channels, n_classes=1)
        '''
        self.direction_heads = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64 // feature_scale, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            ) for _ in range(4)
        ])
        
        self.final_direction = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        '''
        self.flip_head = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        self.save_hyperparameters()
        self.example_input_array = torch.rand(1, n_channels, 128, 128)
        self.test_step_outputs = []

    def forward(self, x):
        x1 = self.unet.inc(x)
        x2 = self.unet.down1(x1)
        x3 = self.unet.down2(x2)
        x4 = self.unet.down3(x3)
        center = self.unet.down4(x4)

        flip_prob = self.flip_head(center)
        return flip_prob  

    def training_step(self, batch, batch_idx):
        # 显式检查设备
        print(f"Model device: {next(self.parameters()).device}")
        print(f"Batch device: {batch[0].device}")
        images, flips = batch
        logits = self(images)
        loss = F.binary_cross_entropy_with_logits(logits.squeeze(), flips.float())
        
        # 打印梯度范数
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                #print(f"梯度范数 {name}: {grad_norm:.4f}")
                #if grad_norm > 1e4:
                    #print(f"警告: 梯度异常 {name}!")
        
        return loss

    def validation_step(self, batch, batch_idx):
        images, flips = batch
        logits = self(images)
        flip_loss = F.binary_cross_entropy_with_logits(logits.squeeze(), flips.float())
        self.log("val_flip_loss", flip_loss, prog_bar=True, on_epoch=True)
        return flip_loss

    def test_step(self, batch, batch_idx):
        images, flips = batch
        logits = self(images)
        preds = torch.sigmoid(logits)
        acc = ((preds > 0.5).float().squeeze() == flips).float().mean()
        self.test_step_outputs.append({"flip_acc": acc})
        self.log("test_flip_acc", acc, prog_bar=True)
        return acc
    
    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            print("Warning: No test outputs collected!")
            return
        avg_acc = torch.stack([x["flip_acc"] for x in self.test_step_outputs]).mean()
        self.log("test_flip_acc_avg", avg_acc)
        print(f"\nFinal Test Accuracy: {avg_acc.item()*100:.2f}%")
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min',
                factor=0.1,
                patience=5,
            ),
            'monitor': 'val_flip_loss',
            'interval': 'epoch'
        }
        return [optimizer], [scheduler]

class DebugCallback(L.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        with torch.no_grad():
            sample = next(iter(train_loader))
            logits = pl_module(sample[0])
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds.squeeze() == sample[1]).float().mean()
            print(f"\nTrain sample accuracy: {acc.item()*100:.2f}%")


# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据集类（简化版，不需要CSV文件）
class RandomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None, num_samples=100):
        """
        参数:
            image_dir (string): 图像文件夹路径
            transform (callable, optional): 可选的变换操作
            num_samples (int): 随机选取的图片数量
        """
        self.image_dir = image_dir
        self.transform = transform
        self.num_samples = num_samples
        
        # 获取文件夹中所有图片文件
        self.image_files = [f for f in os.listdir(image_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # 随机选取指定数量的图片
        if len(self.image_files) > num_samples:
            self.image_files = random.sample(self.image_files, num_samples)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"无法加载图像: {img_path}")
            return None, img_name
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

# 定义图像变换
class mip_downscale:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        w, h = img.size
        while max(w, h) > 2 * 128:
            w, h = w // 2, h // 2
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        return img.resize((128, 128), Image.Resampling.LANCZOS)

def remove_alpha_channel(x):
    if isinstance(x, torch.Tensor):
        return x[:3] if x.shape[0] == 4 else x
    elif isinstance(x, Image.Image):  # PIL Image
        return x.convert('RGB') if x.mode == 'RGBA' else x
    raise TypeError(f"Input must be Tensor or PIL Image. Got {type(x)}")

transform = transforms.Compose([
    transforms.Lambda(remove_alpha_channel),
    mip_downscale(),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def load_model(model_path, num_classes=2):
    """加载预训练的UNet模型"""
    # 重新训练
    '''
    model = DirectionDetectionUNet()  
    model.load_state_dict(torch.load(model_path, map_location=device))
    '''
    # 直接加载模型并预测
    model = DirectionDetectionUNet(n_channels=1, feature_scale=2) 

    state_dict = torch.load(model_path, map_location=device)
    # 处理可能的权重键名不匹配问题
    if any(k.startswith('module.') for k in state_dict):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model

def predict_random_images(model, dataloader):
    """进行预测并返回结果"""
    results = []
    with torch.no_grad():
        for images, img_names in dataloader:
            if images is None:  # 跳过无法加载的图像
                continue
                
            images = images.to(device)
            outputs = model(images)
            print(outputs)
            predicted = torch.round(outputs.data)
            
            for i in range(len(images)):
                results.append({
                    'image_name': img_names[i],
                    'predicted_label': predicted[i].item()
                })
    return results

def main(image_dir, model_path, num_samples=100):
    # 加载数据集
    dataset = RandomImageDataset(image_dir, transform=transform, num_samples=num_samples)
    
    # 过滤掉无法加载的图像
    valid_indices = [i for i in range(len(dataset)) if dataset[i][0] is not None]
    dataset = torch.utils.data.Subset(dataset, valid_indices)
    
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    # 加载模型
    model = load_model(model_path)
    
    # 进行预测
    predictions = predict_random_images(model, dataloader)
    
    # 输出预测结果
    print(f"\n随机选取 {len(predictions)} 张图片的预测结果:")
    print("="*60)
    print(f"{'图像名称':<30}{'预测标签':<15}{'预测标签':<10}")
    print("-"*60)
    
    for pred in predictions:
        label_name = "正向" if pred['predicted_label'] == 0 else "反向"  
        print(f"{pred['image_name']:<30}{label_name:<15}{pred['predicted_label']:.4f}")

main(image_dir='png', model_path='unet_direction_detection_a.pth', num_samples=100)

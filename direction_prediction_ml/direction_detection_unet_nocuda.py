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

class mip_downscale:
    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)
        
        w, h = img.size
        while max(w, h) > 2 * 256:
            w, h = w//2, h//2
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        
        return img.resize((256, 256), Image.Resampling.LANCZOS)
    
def remove_alpha_channel(x):
    return x[:3] if x.shape[0] == 4 else x

transform = transforms.Compose([
    transforms.Lambda(remove_alpha_channel),
    transforms.ToPILImage(),
    mip_downscale(),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def collate_fn(batch):
    images = torch.cat([item[0] for item in batch])
    flips = torch.cat([item[1] for item in batch])
    return images, flips

# Dataset preparation

# --不按fish_id分组---

full_dataset = FishImageDataset(
    annotations_file="fish_direction_detection_results_a.csv",
    img_dir="png",
    transform=transform
)

# 划分数据集
# 假设 full_dataset 是 FishImageDataset 实例，df 是原始 DataFrame
indices = np.arange(len(full_dataset.df))
# train:validation:test=8:1:1
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

# 创建子集
train_subset = torch.utils.data.Subset(full_dataset, train_idx)
val_subset = torch.utils.data.Subset(full_dataset, val_idx)
test_subset = torch.utils.data.Subset(full_dataset, test_idx)

# 创建 DataLoader
train_loader = DataLoader(train_subset, batch_size=32, shuffle=False,pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=32)
test_loader = DataLoader(test_subset, batch_size=32)
## ----

full_dataset = GroupedFishDataset(
    annotations_file="fish_direction_detection_results_a.csv",
    img_dir="png",
    transform=transform
)
print("Train dataset samples:", len(train_subset))
print("Train loader batches:", len(train_loader))
# --按fish_id分组(数据分组过少)---
'''
fish_ids = full_dataset.df['fish_id'].unique()
train_ids, val_ids, test_ids = random_split(fish_ids, [3/6, 1/6, 2/6])

print("Number of train fish IDs:", len(train_ids))
print("Number of val fish IDs:", len(val_ids))
print("Number of test fish IDs:", len(test_ids))

train_dataset = Subset(full_dataset, [i for i, (id,_) in enumerate(full_dataset.grouped) if id in train_ids])
val_dataset = Subset(full_dataset, [i for i, (id,_) in enumerate(full_dataset.grouped) if id in val_ids])
test_dataset = Subset(full_dataset, [i for i, (id,_) in enumerate(full_dataset.grouped) if id in test_ids])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn, num_workers=4)
print("Train dataset samples:", len(train_dataset))
print("Train loader batches:", len(train_loader))
'''
class DeviceDebugCallback(L.Callback):
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # 检查所有tensor设备
        print(f"\nRank {trainer.global_rank} device check:")
        print(f"Model device: {next(pl_module.parameters()).device}")
        for i, t in enumerate(batch):
            if torch.is_tensor(t):
                print(f"Batch tensor {i} device: {t.device}")
            else:
                print(f"Batch item {i} is not tensor")
# Model and training configuration
model = DirectionDetectionUNet(n_channels=1, feature_scale=2)

trainer = L.Trainer(
    strategy='ddp_find_unused_parameters_true',  # 明确使用DDP策略
    accelerator="gpu",
    devices=4,  # 明确指定GPU数量
    precision="16-mixed",  # 使用混合精度
    gradient_clip_val=1.0,  # 防止梯度爆炸
    callbacks=[
        DeviceDebugCallback(),  # 添加设备调试回调
        ModelCheckpoint(monitor="val_flip_loss", save_top_k=1),
        EarlyStopping(
        monitor="val_flip_loss",
        patience=3,  # 3次验证不改善则停止
        mode="min",
        strict=True
    )
    ]
)

# Training and testing
print(f"Actual max_epochs: {trainer.max_epochs}")
trainer.fit(model, train_loader, val_loader)

def test_data_sanity_check():
    test_sample = next(iter(test_loader))
    print("Batch shapes:", [t.shape for t in test_sample])
    
    _, flips = test_sample
    print("Flip labels distribution:", torch.bincount(flips.long()))
    
    model.eval()
    with torch.no_grad():
        logits = model(test_sample[0][:1])
        print("Raw logit value:", logits.item())
        
test_data_sanity_check()

trainer.test(model, test_loader)

# Save model
torch.save(model.state_dict(), "unet_direction_detection_b.pth")

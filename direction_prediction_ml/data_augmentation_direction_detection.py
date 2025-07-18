import os
import random
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import skimage
from skimage.transform import rescale

# 输入annotation
image_dir = 'png'
input_file = 'fish_direction_detection_results.csv'
input_csv = pd.read_csv(input_file)
grouped = list(input_csv.groupby('fish_id'))
num_fish = 6  # 有六条鱼
per_trans = 0.5  # 改变一个fish_id的fish中per_trans比例的fish

def translate_image(image, x_offset, y_offset):
    """
    平移图像（超出部分裁剪）
    :param image: PIL.Image 对象
    :param x_offset: 水平平移像素（正数向右）
    :param y_offset: 垂直平移像素（正数向下）
    :return: 平移后的新图像
    """
    return image.transform(
        image.size,
        Image.AFFINE,
        (1, 0, -x_offset, 0, 1, -y_offset)
    )

# 创建新的DataFrame来存储增强后的数据
augmented_data = input_csv.copy()

for i in range(num_fish):
    fish_id, group_df = grouped[i]
    num_trans = int(len(group_df) * per_trans)
    
    if num_trans > 0:
        # 将pandas Series转换为列表
        image_files = group_df['filename'].tolist()
        
        # 随机选per_trans比例的图像的文件名
        image_files_trans = random.sample(image_files, num_trans)
        
        for image_file_trans in image_files_trans:
            # 读取图像image_file_trans
            img_path = os.path.join(image_dir, image_file_trans)
            try:
                img = Image.open(img_path).convert('RGB')
            except:
                print(f"无法加载图像: {img_path}")
                continue

            img_path = os.path.join(image_dir, image_file_trans)
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
            except Exception as e:
                print(f"无法加载图像 {img_path}: {str(e)}")
                continue
                
            # 获取原始flip值
            original_flip = group_df.loc[group_df['filename'] == image_file_trans, 'flip'].values[0]
            
            # 定义图像变换
            r = random.choice([ -2, -1, 0, 1, 2, 3])
            if r <= 0:
                # 翻转
                new_img = img.transpose(Image.FLIP_TOP_BOTTOM)
                new_flip = 1 - original_flip  # 翻转标签
            elif r == 1:
                # 放大 (使用channel_axis替代multichannel)
                new_img_array = rescale(img_array, scale=1.4, mode='constant', 
                                    channel_axis=2, preserve_range=True)
                new_img = Image.fromarray(new_img_array.astype(np.uint8))
                new_flip = original_flip
            elif r == 2:
                # 缩小
                new_img_array = rescale(img_array, scale=0.7, mode='constant',
                                    channel_axis=2, preserve_range=True)
                new_img = Image.fromarray(new_img_array.astype(np.uint8))
                new_flip = original_flip
            elif r == 3:
                # 平移
                new_img = translate_image(img, 20, 20)
                new_flip = original_flip
            else:
                # 默认水平翻转
                new_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                new_flip = 1 - original_flip

            # 生成新文件名
            new_filename = f"augmented_{image_file_trans}"
            new_img_path = os.path.join(image_dir, new_filename)
            new_img.save(new_img_path)
            
            # 添加新行数据
            new_row = {
                'filename': new_filename,
                'flip': int(new_flip),
                'fish_id': fish_id,
                'angle': group_df.loc[group_df['filename'] == image_file_trans, 'angle'].values[0]
            }
            
            # 添加到增强数据
            augmented_data = augmented_data._append(new_row, ignore_index=True)

# 保存到CSV
augmented_data.to_csv('fish_direction_detection_results_a.csv', index=False)
print(f"数据增强完成，共 {len(augmented_data)} 条数据")

#!/usr/bin/env python3
# 从CIFAR-10的batch1-5提取RGB图像数据集并归一化

import os
import numpy as np
import pickle
from PIL import Image

# 配置
BATCH_PATHS = [
    "cifar-10-batches-py/test_batch"
]  # CIFAR-10的5个训练批次文件
DST_DIR = "data/image_test_batches"   # 目标保存目录
IMAGE_SIZE = 32                  # CIFAR-10图像本身就是32x32，这里仅做确认

os.makedirs(DST_DIR, exist_ok=True)

def load_cifar_batch(file_path):
    """加载CIFAR-10的单个批次文件"""
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')  # CIFAR-10使用latin1编码
    data = batch['data']  # 形状为[N, 3072]，其中3072=32*32*3（RGB）
    labels = batch['labels']  # 标签（这里用不到，但可以保留）
    return data, labels

# 收集所有图像数据
all_images = []
for batch_path in BATCH_PATHS:
    if not os.path.exists(batch_path):
        raise FileNotFoundError(f"CIFAR-10批次文件不存在: {batch_path}")
    # 加载批次数据
    data, _ = load_cifar_batch(batch_path)
    all_images.append(data)
    print(f"已加载 {batch_path}，包含 {len(data)} 张图像")

# 合并所有批次（总共有50000张图像）
all_images = np.concatenate(all_images, axis=0)
print(f"所有批次合并完成，共 {len(all_images)} 张图像")

# 计算全局最大值（CIFAR-10像素值范围是0-255）
global_max = 255.0
print(f"全局最大值: {global_max}（CIFAR-10像素范围）")

# 处理并保存所有图像
for i in range(len(all_images)):
    try:
        # 提取单张图像数据（3072维）并reshape为[3,32,32]
        img_data = all_images[i]  # [3072]
        img_data = img_data.reshape(3, 32, 32)  # [3,32,32]
        # 转换为[H,W,3]格式（PIL需要这种格式）
        img_data = img_data.transpose(1, 2, 0)  # [32,32,3]
        # 转为图像对象（确保数据类型正确）
        img = Image.fromarray(img_data.astype(np.uint8))
        # 归一化到[0,1]
        arr = np.array(img, dtype=np.float32) / global_max
        # 保存为npy
        save_path = os.path.join(DST_DIR, f"train_{i:05d}.npy")
        np.save(save_path, arr)
    except Exception as e:
        print(f"处理第 {i} 张图像失败: {e}，跳过")

# 保存全局最大值
np.save(os.path.join(DST_DIR, "global_max.npy"), np.array(global_max))

print(f"✅ 已保存 {len(all_images)} 个样本到 {DST_DIR}")
# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
import pickle
import numpy as np
from PIL import Image
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms

writer = SummaryWriter("images")
# 定义Tudui模型类（与训练时保持一致）
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # 第二个卷积块
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            # 第三个卷积块
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class CIFAR10Dataset(Dataset):
    """自定义CIFAR-10数据集类"""
    def __init__(self, data_file, transform=None):
        with open(data_file, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        
        # 解码字节字符串
        if b'data' in data_dict:
            self.data = data_dict[b'data']
        elif 'data' in data_dict:
            self.data = data_dict['data']
        else:
            raise KeyError("找不到'data'键")
        
        if b'labels' in data_dict:
            self.labels = data_dict[b'labels']
        elif 'labels' in data_dict:
            self.labels = data_dict['labels']
        else:
            raise KeyError("找不到'labels'键")
        
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # CIFAR-10数据格式：(3072,) -> (3, 32, 32)
        image = self.data[idx].reshape(3, 32, 32).transpose(1, 2, 0)  # (32, 32, 3)
        image = Image.fromarray(image.astype(np.uint8))
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def load_test_data(data_path, batch_size=50):
    """加载测试数据"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    dataset = CIFAR10Dataset(data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader, len(dataset)

def evaluate_model(model, dataloader, device):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy, correct, total

def add_images_to_tensorboard(writer, data_path, tag, num_images=10):
    """添加图片到TensorBoard"""
    dataset = CIFAR10Dataset(data_path, transform=transforms.ToTensor())
    for i in range(min(num_images, len(dataset))):
        image, label = dataset[i]
        # 确保图片格式正确 (C, H, W)
        if image.dim() == 3:
            writer.add_image(f"{tag}/image_{i}_label_{label}", image, i)

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径
    masked_data_path = "./test_sets_100/masked_test_batch_100"
    completed_data_path = "./test_sets_100/completed_test_batch_100"
    test_data_path = "./test_sets_100/test_batch"
    original_data_path = "./test_sets_100/original_test_batch_100"
    # 加载模型
    try:
        # 方法1：使用weights_only=False（简单但安全性较低）
        model = torch.load("tudui2_99.pth", map_location=device, weights_only=False)
        print("模型加载成功")
        
        
    except FileNotFoundError:
        print("模型文件 'tudui2_99.pth' 未找到，请检查文件路径")
        return
    
    #加载图片
    add_images_to_tensorboard(writer, masked_data_path, "masked_data", 10)
    add_images_to_tensorboard(writer, completed_data_path, "completed_data", 10)
    add_images_to_tensorboard(writer, original_data_path, "original_data", 10)
    add_images_to_tensorboard(writer, test_data_path, "original_data", 10)
    # 测试masked数据
    print("\n=== 测试masked数据 ===")
    try:
        masked_loader, masked_total = load_test_data(masked_data_path)
        masked_accuracy, masked_correct, masked_total = evaluate_model(model, masked_loader, device)
        print(f"Masked数据准确率: {masked_accuracy:.2f}% ({masked_correct}/{masked_total})")
    except Exception as e:
        print(f"加载masked数据时出错: {e}")
    
    # 测试completed数据
    print("\n=== 测试completed数据 ===")
    try:
        completed_loader, completed_total = load_test_data(completed_data_path)
        completed_accuracy, completed_correct, completed_total = evaluate_model(model, completed_loader, device)
        print(f"Completed数据准确率: {completed_accuracy:.2f}% ({completed_correct}/{completed_total})")
    except Exception as e:
        print(f"加载completed数据时出错: {e}")
    
    #测试原始数据
    print("\n=== 测试original数据 ===")
    try:
        original_loader, original_total = load_test_data(original_data_path)
        original_accuracy, original_correct, original_total = evaluate_model(model, original_loader, device)
        print(f"Original数据准确率: {original_accuracy:.2f}% ({original_correct}/{original_total})")
    except Exception as e:
        print(f"加载original数据时出错: {e}")


    #测试所有test数据
    print("\n=== 测试test数据 ===")
    try:
        tested_loader, tested_total = load_test_data(test_data_path)
        tested_accuracy, tested_correct, tested_total = evaluate_model(model, tested_loader, device)
        print(f"Test数据准确率: {tested_accuracy:.2f}% ({tested_correct}/{tested_total})")
    except Exception as e:
        print(f"加载test数据时出错: {e}")

if __name__ == "__main__":
    main()
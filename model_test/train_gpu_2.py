# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
#这是在GPU上训练的代码，使用GPU训练，需要将模型、数据集、损失函数、优化器都放到GPU上
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

# from model import *
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

# 定义训练的设备，需要将模型、损失函数、图像、标签加载到GPU上
print(torch.cuda.is_available())
device = torch.device("cuda")

# 数据增强
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = torchvision.datasets.CIFAR10(root="dataset1", train=True, transform=train_transform,
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="dataset1", train=False, transform=test_transform,
                                         download=True)

# length 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
# 如果train_data_size=10, 训练数据集的长度为：10
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))


# 利用 DataLoader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=6, pin_memory=True)
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=True, num_workers=6, pin_memory=True)

# 创建网络模型
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
tudui = Tudui()
tudui = tudui.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
loss_fn = loss_fn.to(device)
# 优化器
# learning_rate = 0.01
learning_rate = 1e-2
optimizer = torch.optim.AdamW(tudui.parameters(), lr=learning_rate, weight_decay=1e-4)
# 学习率调度
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, min_lr=1e-5)
# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 200
# 早停参数
best_accuracy = 0.0
patience = 10
patience_counter = 0

# 添加tensorboard
writer = SummaryWriter("logs_train3")

for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 训练步骤开始
    total_train_loss = 0
    total_train_accuracy = 0
    tudui.train()
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_loss = total_train_loss + loss.item()
        total_train_accuracy = total_train_accuracy + (outputs.argmax(1) == targets).sum()
        total_train_step = total_train_step + 1
    writer.add_scalar("train_loss", total_train_loss/train_data_size, i)
    writer.add_scalar("train_accuracy", total_train_accuracy/train_data_size, i)
    print("整体训练集上的Loss: {}".format(total_train_loss/train_data_size))
    print("整体训练集上的正确率: {}".format(total_train_accuracy/train_data_size))

    # 测试步骤开始
    tudui.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_test_accuracy = total_test_accuracy + accuracy

    print("整体测试集上的Loss: {}".format(total_test_loss/test_data_size))
    print("整体测试集上的正确率: {}".format(total_test_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss/test_data_size, i)
    writer.add_scalar("test_accuracy", total_test_accuracy/test_data_size, i)
    total_test_step = total_test_step + 1
    print("当前学习率: {}".format(optimizer.param_groups[0]['lr']))
    writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], i)
    scheduler.step(total_test_loss/test_data_size)
    # 早停逻辑
    if total_test_accuracy > best_accuracy:
        best_accuracy = total_test_accuracy
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print("早停触发！")
        break
    #torch.save(tudui, "tudui_{}.pth".format(i))
    #print("模型已保存")
torch.save(tudui, "tudui3_{}.pth".format(i))
writer.close()

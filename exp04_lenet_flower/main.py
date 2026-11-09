import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   # 数据目录以脚本所在位置为基准，换机器不用改路径
DATA_ROOT = BASE_DIR / "Data"

#先搭lenet代码框架
import  torch
import  torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('TkAgg')   # 切换后端
import matplotlib.pyplot as plt

import time
class MyLeNet(nn.Module):
    def __init__(self,num_classes):
        ''''
        #根据课上的讲解，LeNet由卷积-池化-卷积-全连接-全连接-全连接构成
        #定义卷积层:卷积-池化-卷积-池化
        '''
        super().__init__()
        self.features = nn.Sequential(
            #第一个卷积核 输入channel=3 输出channel=6 卷积核5*5
            nn.Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1), padding=0),
            nn.ReLU(),#激活函数
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),#池化，对输入减半
            nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        #定义全连接层
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),#输入是上一个卷积核
            nn.ReLU(),#激活函数
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),#经过120-84-num_classes之后输出对物品的分类
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)#全脸接层只接受二维输入所以只能展平成一维数据
        x = self.classifier(x)
        return x
#================训练环节=================
#======超参数定义=====
batch_size = 16
lr=0.001
epochs = 30
num_classes = 5#测试集与训练集都是五种花
#=====下载数据集，进行数据增强。=====
#数据增强定义，这一部分代码是又大模型给出。自己添加注释
train_transform = transforms.Compose([
    transforms.Resize((32,32)),#将图片大小改为32*32适配LeNet
    transforms.RandomHorizontalFlip(0.5),#翻转率大小
    transforms.RandomRotation(10),#随机旋转度数-10-10
    transforms.ToTensor(),#转化为张量
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])#标准化
])
#测试集去掉翻转与旋转，确保真实输出
val_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# 加载数据集

train_dataset = datasets.ImageFolder(root=str(DATA_ROOT / "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(root=str(DATA_ROOT / "test"), transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 保存训练记录
train_loss_list, train_acc_list = [], []
val_loss_list, val_acc_list = [], []
# 模型、设备、损失、优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#定义设备，如果cuda可用就用cuda否则就用cpu
model = MyLeNet(num_classes).to(device)#加载模型到设备
criterion = nn.CrossEntropyLoss()#实例化损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=lr)#实例化优化器
for epoch in range(epochs):#在一定轮次下进行训练与验证
    start_time = time.time()#记录开始时间
    # 训练阶段
    model.train()#调用模型训练
    running_loss = 0.0#初始化损失值
    correct = 0#初始化正确值
    total = 0#初始化测试总数
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()#保存梯度
        outputs = model(imgs)
        loss = criterion(outputs, labels)#计算损失值
        loss.backward()#反向传播
        optimizer.step()#根据梯度更新权重

        running_loss += loss.item()
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total#计算准确率

    # 验证阶段
    #验证阶段的代码和训练阶段类似，但是不用更新梯度，因为这是测试
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, pred = torch.max(outputs.data,1)
            val_total += labels.size(0)
            val_correct += (pred == labels).sum().item()
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    print(f"Epoch [{epoch+1}/{epochs}] Time:{time.time()-start_time:.2f}s")
    print(f"Train Loss:{train_loss:.4f} Acc:{train_acc:.2f}% | Val Loss:{val_loss:.4f} Acc:{val_acc:.2f}%")

# 绘制训练曲线
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1,2,2)
plt.plot(train_acc_list, label="Train Acc")
plt.plot(val_acc_list, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy(%)")
plt.legend()
plt.title("Accuracy Curve")
plt.tight_layout()
plt.show()

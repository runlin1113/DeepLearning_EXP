import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   # 数据目录以脚本所在位置为基准，换机器不用改路径
DATA_ROOT = BASE_DIR / "Data"

#先搭lenet代码框架
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time

class MyLeNet(nn.Module):
    def __init__(self,num_classes):
        '''
        根据课上的讲解，LeNet由卷积-池化-卷积-池化-全连接-全连接-全连接构成
        定义卷积特征提取层:卷积-池化-卷积-池化
        '''
        super().__init__()
        self.features = nn.Sequential(
            #第一个卷积层 输入channel=3 输出channel=6 卷积核5*5
            nn.Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1), padding=0),
            nn.ReLU(),#激活函数
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),#池化，尺寸减半
            nn.Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1), padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        #定义全连接分类层
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),#输入特征
            nn.ReLU(),#激活函数
            nn.Dropout(0.2),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, num_classes),#输出各类得分
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)#展平，batch保留，其余一维
        x = self.classifier(x)
        return x

#================训练环节=================
#======超参数定义=====
batch_size = 16
lr=0.001
epochs = 30
num_classes = 5#5种花分类
SAVE_ACC_THRESHOLD = 95.0  # 设置保存阈值 >=95%就保存
saved_flag = False          # 标记：是否已经保存过满足条件的模型

#=====数据增强=====
train_transform = transforms.Compose([
    transforms.Resize((32,32)),#图片resize到32*32适配LeNet
    transforms.RandomHorizontalFlip(0.5),#随机水平翻转
    transforms.RandomRotation(10),#随机旋转±10度
    transforms.ToTensor(),#转张量，0~1
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])#ImageNet标准化
])
#测试集不用随机增强
val_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# 加载数据集（Data/ 与脚本同级）
train_dataset = datasets.ImageFolder(root=str(DATA_ROOT / "train"), transform=train_transform)
val_dataset = datasets.ImageFolder(root=str(DATA_ROOT / "test"), transform=val_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 保存训练记录
train_loss_list, train_acc_list = [], []
val_loss_list, val_acc_list = [], []

# 模型、设备、损失、优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyLeNet(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(epochs):
    start_time = time.time()
    # 训练阶段
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    # 验证阶段
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

    # ========== 判断验证集acc>=95% 保存模型 + 提前停止 ==========
    if not saved_flag and val_acc >= SAVE_ACC_THRESHOLD:
        save_name = f"lenet_valacc95_epoch{epoch+1}_acc{val_acc:.2f}.pth"
        torch.save(model.state_dict(), save_name)
        print(f"\n✅ 验证准确率达到 {val_acc:.2f}%，保存模型为 {save_name}")
        saved_flag = True
        print("🎯 目标准确率达成，提前终止训练！")
        break   #跳出epoch循环

# 训练全部跑完后绘图
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
plt.savefig("train_curve.png", dpi=300)
print("\n训练曲线图保存为 train_curve.png")

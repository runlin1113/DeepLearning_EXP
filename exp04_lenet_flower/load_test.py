# load_test.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   # 数据目录以脚本所在位置为基准，换机器不用改路径
DATA_ROOT = BASE_DIR / "Data"

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import random

# =========【和训练时完全一致！没有Dropout！】=========
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

# ===================== 配置 =====================
num_classes = 5
# 权重自动查找：优先用目录里最新的 lenet_*.pth（权重文件与本脚本同级）
_cands = sorted(BASE_DIR.glob("lenet_*.pth"), key=lambda x: x.stat().st_mtime, reverse=True)
MODEL_PATH = str(_cands[0]) if _cands else str(BASE_DIR / "lenet_valacc95_epoch29_acc95.00.pth")

val_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# 测试集完整路径
test_root = str(DATA_ROOT / "test")      # 测试集与本脚本同级的 Data/test
test_dataset = datasets.ImageFolder(root=test_root, transform=val_transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 标签映射：数字→类别名
idx_to_class = {v:k for k, v in test_dataset.class_to_idx.items()}
print("类别映射：", idx_to_class)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyLeNet(num_classes).to(device)

#加载模型权重
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("\n模型加载成功！")

#==== 1. 在整个独立测试集上评估总准确率====
correct = 0
total = 0
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

test_acc = 100.0 * correct / total
print(f"\n【独立测试集评估】测试集准确率 = {test_acc:.2f} %")

#==== 2. 随机抽取5个样本打印预测结果（用于截图放进报告）====
print("\n====随机抽样预测样例====")
all_samples = list(test_dataset)
random.seed(42) #固定随机种子，每次运行抽到一样图片，方便写报告
sample_list = random.sample(all_samples,5)

with torch.no_grad():
    for idx, (img, label) in enumerate(sample_list):
        img_input = img.unsqueeze(0).to(device) #增加batch维度
        output = model(img_input)
        _, pred = torch.max(output, dim=1)
        true_name = idx_to_class[label]
        pred_name = idx_to_class[pred.item()]
        print(f"样本{idx+1} | 真实类别：{true_name} | 预测类别：{pred_name}")

#====【可选：单张图片预测函数，放在最后】====
from PIL import Image
def predict_single_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = val_transform(img)
    img_input = img.unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_input)
        _, pred = torch.max(out, dim=1)
    return idx_to_class[pred.item()]

#调用示例：把下面这行注释取消，填入图片路径即可
# res = predict_single_image(r"D:\xxx\xxx.jpg")
# print("单张图片预测结果：", res)

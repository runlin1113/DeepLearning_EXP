# 实验四 · LeNet-5 五类花卉图像分类

按课上讲的 LeNet 结构手写一遍（不调用 `torchvision.models`），在 5 类花卉的极小数据集上跑通「训练 → 保存 → 加载 → 单张推理」的完整闭环。

## 数据

```
Data/
├── train/            75 张
│   ├── GuiZhenCao 16   JiaLianQiao 16   PaShanHu 14
│   └── YeZiHua    15   YinHeHuan  14
└── test/             20 张（每类 4 张）
```

类别是 5 种中草药花卉。数据量极小（95 张），所以这份代码的重点不是精度，而是**把 LeNet 的每一层形状算对**。

## 文件

| 文件 | 说明 |
|---|---|
| `exp.py` | 完整训练脚本，分类头带 `Dropout(0.2)`，验证集 ≥95% 才存权重 |
| `main.py` | 同一网络的另一版：不带 Dropout，`matplotlib` 用 `TkAgg` 实时画曲线 |
| `load_test.py` | 加载权重做独立测试集评估 + 随机抽样预测；**权重文件自动查找** |
| `lenet_*.pth` | 已训练权重（val_acc 95.00%），两个 epoch 各存一份 |

`exp.py` 与 `main.py` 是有意保留的两次迭代：前者加了 Dropout 和「达标才存盘」，后者是更早的骨架版本。对比着看能看出正则化与小数据集过拟合的关系。

## 运行

```bash
pip install torch torchvision matplotlib
python exp.py          # 训练 30 epoch，达标自动存 lenet_valacc*_acc*.pth
python load_test.py    # 独立测试集评估（权重自动选最新的）
python main.py         # 训练并实时绘制 loss/acc 曲线
```

CPU 上一轮几秒，整轮训练 1~2 分钟。

实测输出：

```
模型加载成功！
【独立测试集评估】测试集准确率 = 95.00 %
```

## 网络结构

```
输入 3×32×32
 Conv2d(3→6, 5×5)  → ReLU → MaxPool(2)     # 6×14×14
 Conv2d(6→16, 5×5) → ReLU → MaxPool(2)     # 16×5×5
 Flatten                                    # 400
 Linear(400→120) → ReLU → Dropout(0.2)
 Linear(120→84)  → ReLU
 Linear(84→5)
```

两处与原版 LeNet 的差异，都是为了让它在 32×32 彩色输入上跑通：

1. 第一层输入通道改成 **3**（原版是单通道 32×32 手写数字）；
2. 全连接输入是 **16×5×5=400**，不是原版的 16×5×5 之外的 120——因为 32×32 经过两次池化正好到 5×5，与 MNIST 的 28×28 不同。

## 训练设置

| 项 | 值 |
|---|---|
| batch size | 16 |
| 优化器 | Adam, lr = 1e-3 |
| epoch | 30 |
| 增强 | RandomHorizontalFlip(0.5) + RandomRotation(±10°) + ImageNet 标准化 |
| 存盘条件 | 验证准确率 ≥ 95% |

## 容易踩的坑

- **测试集不能加随机增强**（翻转/旋转），否则每次评估结果都不一样；
- `load_state_dict` 报 `Missing key(s) ... dropout` 不是错误：Dropout 没有参数，`strict=True` 也能过；
- 权重路径不要写死。`load_test.py` 用 `sorted(BASE_DIR.glob("lenet_*.pth"), key=mtime)` 自动挑最新的一份，换机器也能跑。

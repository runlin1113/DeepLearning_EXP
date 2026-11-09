# 实验五 · 语音唤醒词识别（Mel 频谱 + 轻量 CNN）

从原始录音到「说出唤醒词就亮灯」的完整链路：**m4a 转 wav → Mel 频谱图 → 固定尺寸张量 → CNN 分类 → 单文件推理**。

数据是自采的，因此这份工程的看点是**小样本（每类几十条）下怎么不过拟合**：固定时长裁剪、SpecAugment、早停、类别不均衡处理。

## 流程

```
dataset/<类别>/raw/*.m4a
    │ convert_m4a_to_wav.py      ffmpeg 转码 + 统一采样率
    ▼
dataset/<类别>/wav_out/*.wav
    │ preprocess_audio.py        librosa → 40 维 Mel → dB → 截断/补零到 32 帧
    ▼
preprocessed/mel_npy/*.npy        (40, 32) float32    + mel_images/*.png（人工检查用）
    │ train_cnn.py               SpecAugment + 5 折早停
    ▼
model_output/wakeword_cnn.pt      {'model_state', 'class_names', 'mean', 'std'}
    │ infer.py
    ▼
  类别 + 概率
```

## 文件

| 文件 | 作用 |
|---|---|
| `convert_m4a_to_wav.py` | 批量转码，ffmpeg 路径优先取环境变量 `FFMPEG_EXE`，找不到再查 PATH |
| `preprocess_audio.py` | Mel 频谱提取，产出 `.npy` 与 `.png` 两份（后者用来肉眼确认频谱没画错） |
| `train_cnn.py` | 训练主脚本：划分、增强、早停、保存最优权重与归一化参数 |
| `infer.py` | 单文件推理：`python infer.py xxx.wav` |
| `model_output/` | 已训练权重 `wakeword_cnn.pt` + 训练曲线 + 混淆矩阵 |
| `assets/` | 架构图与流程图 |

## 数据准备

仓库不含录音（个人语音数据）。按下面结构放好自己的数据即可：

```
dataset/
├── laicai/raw/*.m4a        唤醒词 1
├── qidong/raw/*.m4a        唤醒词 2
├── self/raw/*.m4a          呼名（叫自己的名字）
└── other/raw/*.m4a         负类：任意其它说话内容
```

实验里共 81 条（4 类，训练 65 / 验证 16）。每类 20 条以上就能跑，越多越好。`other` 类是**必须的**——只训唤醒词、不给负样本，模型会把任何声音都判成唤醒词。

## 运行

```bash
pip install librosa soundfile numpy matplotlib torch
python convert_m4a_to_wav.py     # 需要 ffmpeg
python preprocess_audio.py
python train_cnn.py              # 120 epoch，验证集 20%，早停 patience=20
python infer.py dataset/laicai/wav_out/laicai_001.wav
```

只想试推理的话，`model_output/wakeword_cnn.pt` 已上传，直接跑最后一条命令即可（需要一个 wav）。

## 模型

```
输入 (1, 40, 32)
Conv(1→16, 3×3) → ReLU → MaxPool2      # 16×20×16
Conv(16→32, 3×3) → ReLU → MaxPool2     # 32×10×8
Conv(32→64, 3×3) → ReLU → MaxPool2     # 64×5×4
Flatten → 1280
Linear(1280→64) → ReLU → Dropout(0.5)
Linear(64→4)
```

约 8.7 万参数。对 40×32 的输入来说，三次池化后只剩 5×4，再深就会把时频结构压没，所以到第三层就停。

## 关键设置与理由

| 设置 | 值 | 为什么 |
|---|---|---|
| 采样率 | 16 kHz | 语音识别通用；再高对唤醒词无增益，只会让 Mel 计算变慢 |
| Mel 维度 | 40 | 覆盖 0~8 kHz，与 16 kHz 采样率匹配（Nyquist = 8 kHz） |
| 时间帧 | 32 | 固定尺寸才能进 CNN；不足补零、超出截断 |
| 归一化 | `(x - mean) / std`，均值 -51.4 / 标准差 18.2 | **这两个数随权重一起存盘**，否则推理时 Mel 的 dB 尺度对不上，概率全错 |
| SpecAugment | 时间遮 5 帧 / 频率遮 3 带 | 小样本下遮太多反而学不到时频结构，强度刻意压低 |
| 早停 | patience 20 | 120 epoch 上限只是保险，实际通常 40~60 轮就停 |

## 踩过的坑

- **dB 尺度不统一**：`power_to_db(ref=np.max)` 是逐条归一的，训练与推理必须都调用同一函数，不能一条用 `ref=1.0`、一条用 `ref=np.max`；
- **截断 vs 补零**：录音长短不一时，前半段往往还没说完整，简单截前 32 帧可能把关键词切掉；实际用法是「先按能量做 VAD，再从起点切」；
- **权重里的 `class_names` 顺序**：`preprocess_audio.py` 按文件夹名字母序排序，`train_cnn.py` 沿用这个顺序，`infer.py` 从 ckpt 里读回顺序——三处必须一致，否则概率对错标签。

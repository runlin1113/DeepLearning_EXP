# 深度学习课程实验代码

深圳技术大学《深度学习》课程实验代码合集。每次实验一个独立目录，**只保留可复现的关键代码**，不含数据集、模型权重与个人数据。

| 目录 | 实验内容 | 核心技术 | 代码量 |
|---|---|---|---|
| [`exp02_audio_basics`](exp02_audio_basics/) | 音频采集与实时波形显示 | sounddevice 回调 + 队列 + matplotlib 定时器 | 2 个脚本 |
| [`exp04_lenet_flower`](exp04_lenet_flower/) | LeNet-5 五类花卉图像分类 | 手写 LeNet（Conv-Pool-Conv-Pool-FC×3） | 3 个脚本 + 95 张图 |
| [`exp05_wakeword_cnn`](exp05_wakeword_cnn/) | 语音唤醒词识别 | Mel 频谱 + 轻量 CNN + SpecAugment | 4 个脚本 + 训练好的权重 |
| [`exp06_yolo11_sprint_start`](exp06_yolo11_sprint_start/) | 起跑动作目标检测 | YOLO11n + aHash 抗泄漏分组交叉验证 | 14 个脚本 |
| [`exp07_face_recognition`](exp07_face_recognition/) | 人脸检测与动态人脸识别 | YuNet + 相似变换对齐 + ArcFace + 多帧追踪 | 工程（约 4400 行） |

实验三、实验七的编号沿用课程原始排课，未出现的编号当时未留代码。

## 快速开始

每个子目录的 README 都有独立的运行说明。通用约定：

```bash
git clone https://github.com/runlin1113/DeepLearning_EXP.git
cd DeepLearning_EXP/exp07_face_recognition
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ 可跑。涉及训练的脚本建议装 CUDA 版 PyTorch；纯推理部分 CPU 即可。

## 关于路径

所有脚本的路径都以**脚本自身所在目录**为基准（`Path(__file__).resolve().parent`，或在脚本入口 `os.chdir` 到脚本目录），因此仓库可以克隆到任意位置直接运行，不需要改任何绝对路径。

YOLO 的 `data.yaml` 里刻意**不写 `path:` 字段**——ultralytics 会退化为「以 yaml 所在目录为数据集根」，正好对应 `exp06/dataset/`，换机器也不会失效。

## 关于数据与权重

本仓库**不上传**原始数据集、模型权重和个人数据：

| 类型 | 处理方式 |
|---|---|
| 自采图像/录音 | 不上传。各 README 说明目录结构，按格式自行放入即可 |
| 公开数据集（LFW 子集） | 不上传，README 给出获取方式 |
| 大模型权重（ArcFace 167 MB） | 不上传，`step0_download_models.py` 自动下载并校验 sha256 |
| 小模型权重（LeNet 251 KB / 唤醒词 CNN 425 KB） | 已上传，方便直接跑推理 |

## 环境版本（实测）

```
Python 3.13 / 3.10      numpy 2.x      opencv-contrib-python 4.12
onnxruntime 1.20+       torch 2.6.0+cu124     ultralytics 8.4
librosa 0.10            sounddevice 0.5       matplotlib 3.9
```

## License

MIT

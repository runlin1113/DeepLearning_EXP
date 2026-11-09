# 实验六 · YOLO11 起跑动作检测（含抗泄漏数据划分）

用 YOLO11n 检测短跑起跑姿势里的 `start` 目标。数据只有 **52 张手机连拍**，这个实验的重点因此不在调模型，而在**怎么在 52 张图上得到可信的评估数字**——随机划分会让连拍近重复图同时落进训练集和验证集，mAP 虚高。

## 核心思路：先按图分组，再划分

```
52 张图 ──aHash 感知哈希──> 汉明距离 ≤ 6 视为近重复 ──> 28 个独立组
                                                        │
        ┌───────────────────────────────────────────────┤
        ▼                        ▼                      ▼
   5 折交叉验证           分组单划分 42/10          全量 52（部署）
   （按组切，组不跨折）    （按组切，可复现）        （train==val，仅看收敛）
```

判据是 **aHash + 汉明距离**：连拍的相邻帧缩放、轻微位移后哈希几乎不变，而不同动作阶段差异大。这一步把「同一秒拍的一串照片」绑在一起，杜绝了最典型的泄漏源。

## 文件

| 文件 | 作用 |
|---|---|
| `make_splits.py` | **入口**：aHash 分组 → 5 折 / 分组划分 / 全量，写 txt 与 `data_*.yaml` |
| `train_yolo.py` | 基线：原随机 42/10 划分，用于对比「泄漏能虚高多少」 |
| `train_kfold.py` | 分组 5 折 + 最终部署模型；读 `results.csv` 取最优轮，避开 Windows 多进程崩溃 |
| `complete_fold0.py` | 断点续跑单折 |
| `retrain_deploy.py` | 用分组划分 / 全量划分重训部署权重 |
| `eval_test.py` | 10 张**从未参与训练**的独立测试集评估 |
| `predict_yolo.py` | 推理：支持单图 / 文件夹 / 视频 |
| `predict_test.py` | 对独立测试集批量出带框图 |
| `read_results.py` / `update_metrics.py` | 从各 run 的 `results.csv` 抽指标，汇总进 `kfold_metrics.json` |
| `analyze_folds.py` | 统计各折验证集的框面积分布，解释方差来源 |
| `make_final_charts.py` | 出三张图：方案对比、折间方差、测试集指标 |
| `prepare_testset.py` / `prepare_test_data.py` | 造独立测试集：从照片池挑 10 张，X-AnyLabeling 的 json 转 YOLO txt |
| `dataset/kfold/groups.json` | 分组结果（**已上传**，可直接查 28 组怎么分的） |
| `dataset/kfold/kfold_metrics.json` | 全部指标（**已上传**） |
| `figures/` | 结果图 |

仓库不含 52 张训练图（自采照片，含个人影像）。`dataset/images/`、`dataset/labels/`、`dataset/photo/` 需自行按 YOLO 格式放置：

```
dataset/
├── images/train/*.jpg      labels/train/*.txt     # 42 张
├── images/val/*.jpg        labels/val/*.txt       # 10 张
└── data.yaml                                     # 只写 train/val/nc/names，不写 path
```

## 运行

```bash
pip install ultralytics
python make_splits.py        # 必须先跑：生成 5 折与分组划分（依赖 images/train + val）
python train_yolo.py         # 基线（随机划分）
python train_kfold.py        # 分组 5 折 + 部署模型
python retrain_deploy.py     # 用分组/全量划分重训
python eval_test.py          # 独立测试集
python predict_yolo.py 某张图.jpg
python make_final_charts.py  # 出图
```

## 结果

分组 K 折（5 折，按组切）：

| 折 | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 0 | 1.000 | 0.238 | 0.302 | 0.221 |
| 1 | 1.000 | 0.858 | 0.971 | 0.791 |
| 2 | 0.648 | 0.500 | 0.455 | 0.295 |
| 3 | 0.369 | 0.375 | 0.327 | 0.207 |
| 4 | 1.000 | 0.990 | 0.995 | 0.741 |
| **均值±标准差** | 0.803±0.257 | 0.592±0.286 | **0.610±0.309** | **0.451±0.259** |

其他三种划分：

| 划分 | mAP50 | mAP50-95 | 说明 |
|---|---|---|---|
| 随机 42/10（泄漏） | 0.99+ | — | 同组图片跨集，指标虚高不可信 |
| 分组 42/10 | 0.995 | 0.842 | 唯一可作泛化估计的单次划分 |
| 全量 52 | 0.995 | 0.900 | train == val，**不是泛化指标** |
| 独立测试 10 张 | 0.968 | 0.625 | 真正的样本外，但只有 10 张，方差大 |

**四组数字放在一起才是结论**：分组后折间标准差高达 0.26，说明在 52 张图上「一个 mAP 数字」没有意义，必须报均值±标准差；而「随机划分 0.99」与「分组划分 0.842」的差距，就是泄漏带来的虚高。

## 踩过的坑

- **Windows 下 `val()` 递归崩溃**：ultralytics 的 dataloader 用 `spawn` 起子进程，若训练脚本没有 `if __name__ == "__main__":`，子进程会重新执行整个脚本 → 无限递归。两个训练脚本都加了主入口保护，且 `workers=0`。
- **指标不要额外调 `val()`**：直接读 run 目录的 `results.csv`，取 mAP50-95 最高那一轮，和 ultralytics 选 `best.pt` 的准则一致，还省一次推理。
- **`data.yaml` 里不要写绝对路径**：本仓库的 yaml 故意省略 `path:` 字段，ultralytics 会以 yaml 所在目录为数据集根，克隆到任何机器都能用。`make_splits.py` 生成的 yaml 会写入本机绝对路径，那是运行时行为，不随仓库传播。
- **10 张测试集太少**：mAP50-95 从 0.90 掉到 0.625 不全是过拟合，也有 10 张样本的抽样噪声。要下结论得扩到 50 张以上。

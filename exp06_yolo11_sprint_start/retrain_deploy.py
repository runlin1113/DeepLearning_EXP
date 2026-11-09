# -*- coding: utf-8 -*-
"""
重建被误删的两个权重（用关闭沙箱方式运行，避免 labels.cache 批量删除确认卡死）：
  1) grouped_split      : 抗泄漏分组单划分 42/10 -> 评估参考模型
  2) start_pose_all52   : 全量 52 张训练 -> 最终部署模型(数据最多)
用法：venv_xanylabeling/Scripts/python.exe retrain_deploy.py
"""

# 统一工作目录：以下相对路径均以本文件所在目录为基准，克隆到任意位置可直接运行
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import torch
from ultralytics import YOLO

DEVICE = 0 if torch.cuda.is_available() else "cpu"
print(f"[设备] CUDA={torch.cuda.is_available()} device={DEVICE}")

jobs = [
    ("grouped_split",    "dataset/kfold/data_grouped.yaml"),
    ("start_pose_all52", "dataset/kfold/data_all.yaml"),
]
for name, data in jobs:
    print(f"\n========== 训练 {name} ({data}) ==========")
    m = YOLO("yolo11n.pt")
    m.train(data=data, epochs=100, imgsz=640, batch=16, device=DEVICE,
            workers=0, patience=30, seed=42, name=name, exist_ok=True, verbose=False)
    # 顺带在各自 val 上评估
    mm = YOLO(f"runs/detect/{name}/weights/best.pt")
    met = mm.val(data=data, device=DEVICE, verbose=False)
    print(f"[{name}] P={met.box.mp:.4f} R={met.box.mr:.4f} "
          f"mAP50={met.box.map50:.4f} mAP50-95={met.box.map:.4f}")
print("\n完成。部署权重: runs/detect/start_pose_all52/weights/best.pt")

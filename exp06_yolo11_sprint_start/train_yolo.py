# -*- coding: utf-8 -*-
"""
实验六：用最新 YOLO 训练「起跑动作(start)」检测器（基线：原随机 42/10 划分）
依赖：ultralytics + CUDA 版 torch
运行：venv_xanylabeling\\Scripts\\activate.bat  ->  python train_yolo.py
说明：有 GPU 自动走 CUDA(device=0)；Windows 下 workers=0；
      已加 if __name__ == "__main__" 主入口保护，避免 val() 的 spawn 递归崩溃。
"""

# 统一工作目录：以下相对路径均以本文件所在目录为基准，克隆到任意位置可直接运行
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import torch
from ultralytics import YOLO


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[设备] CUDA 可用: {torch.cuda.is_available()}  训练设备: {device}")
    if torch.cuda.is_available():
        print(f"[设备] GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO("yolo11n.pt")
    model.train(
        data="dataset/data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=device,
        workers=0,
        name="start_pose_yolo11n",
        patience=30,
    )

    metrics = model.val(device=device)
    print("mAP50-95:", metrics.box.map)
    print("mAP50   :", metrics.box.map50)
    print("Precision:", metrics.box.mp)
    print("Recall   :", metrics.box.mr)


if __name__ == "__main__":
    main()

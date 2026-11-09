# -*- coding: utf-8 -*-
"""用全量部署模型对独立测试集做推理可视化，输出带框图到 runs/detect/predict_test/。"""

# 统一工作目录：以下相对路径均以本文件所在目录为基准，克隆到任意位置可直接运行
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import torch
from ultralytics import YOLO


def main():
    dev = 0 if torch.cuda.is_available() else "cpu"
    m = YOLO("runs/detect/start_pose_all52/weights/best.pt")
    m.predict(source="dataset/test/images", save=True, conf=0.25,
              device=dev, workers=0, name="predict_test")
    print("完成，结果在 runs/detect/predict_test/")


if __name__ == "__main__":
    main()

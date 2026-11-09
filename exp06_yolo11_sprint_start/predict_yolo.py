# -*- coding: utf-8 -*-
"""
实验六：起跑动作(start)推理演示
用法：
  python predict_yolo.py                 # 默认在验证集上推理，保存带框图
  python predict_yolo.py 图片或视频路径   # 对任意新图/视频推理
  python predict_yolo.py 图片文件夹       # 对整批新图推理
输出：runs/detect/predict/  (带检测框的图片/视频)
"""

# 统一工作目录：以下相对路径均以本文件所在目录为基准，克隆到任意位置可直接运行
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import sys
from ultralytics import YOLO

# 默认使用「全量 52 张」训练的最终部署模型；
# 若想对比早期随机划分模型，改成 runs/detect/start_pose_yolo11n/weights/best.pt
WEIGHTS = "runs/detect/start_pose_all52/weights/best.pt"


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "dataset/images/val"
    model = YOLO(WEIGHTS)
    model.predict(source=source, save=True, conf=0.25, device=0)
    print("推理完成，结果保存在 runs/detect/predict/")


if __name__ == "__main__":
    main()

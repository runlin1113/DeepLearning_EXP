# -*- coding: utf-8 -*-
"""
独立测试集评估：用全量部署模型 runs/detect/start_pose_all52 在 dataset/test 上做
真正的「样本外」评估（这 10 张从未参与训练）。
已加主入口保护 + workers=0，规避 Windows 多进程 spawn 崩溃。
"""

# 统一工作目录：以下相对路径均以本文件所在目录为基准，克隆到任意位置可直接运行
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import json
import torch
from ultralytics import YOLO

WEIGHTS = "runs/detect/start_pose_all52/weights/best.pt"
DATA = "dataset/test/data_test.yaml"


def main():
    dev = 0 if torch.cuda.is_available() else "cpu"
    print(f"[设备] device={dev}")
    model = YOLO(WEIGHTS)
    met = model.val(data=DATA, device=dev, workers=0, verbose=True, plots=True)
    out = {
        "precision": round(float(met.box.mp), 4),
        "recall": round(float(met.box.mr), 4),
        "map50": round(float(met.box.map50), 4),
        "map5095": round(float(met.box.map), 4),
    }
    print("\n===== 测试集（10 张，完全样本外）指标 =====")
    print(f"  Precision : {out['precision']}")
    print(f"  Recall    : {out['recall']}")
    print(f"  mAP@0.5   : {out['map50']}")
    print(f"  mAP@0.5:0.95 : {out['map5095']}")
    with open("dataset/test/test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n已写入 dataset/test/test_metrics.json")


if __name__ == "__main__":
    main()

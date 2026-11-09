# -*- coding: utf-8 -*-
"""回填 fold0 + all52 到 kfold_metrics.json，并重算 5 折均值±标准差。
数值来源：各 run 目录 results.csv 的「mAP50-95 最高轮」(即 ultralytics 选 best.pt 的准则)。
"""

# 统一工作目录：以下相对路径均以本文件所在目录为基准，克隆到任意位置可直接运行
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import json

jp = "dataset/kfold/kfold_metrics.json"
d = json.load(open(jp, encoding="utf-8"))

# fold0: best epoch14  P=1.0 R=0.2381 mAP50=0.30171 mAP50-95=0.22097
d["folds"]["0"] = {"precision": 1.0, "recall": 0.2381, "map50": 0.3017, "map5095": 0.221}
# all52: best epoch94  P=0.9975 R=1.0 mAP50=0.995 mAP50-95=0.90033 (train==val, 仅健康度)
d["all52"] = {"precision": 0.9975, "recall": 1.0, "map50": 0.995, "map5095": 0.9003,
              "note": "train==val，非泛化指标；仅表示部署模型在已见数据上健康"}

keys = ["precision", "recall", "map50", "map5095"]
valid = [d["folds"][str(i)] for i in range(5)]
mean, std = {}, {}
for k in keys:
    xs = [v[k] for v in valid]
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    mean[k] = round(m, 4)
    std[k] = round(var ** 0.5, 4)
d["kfold_mean"] = mean
d["kfold_std"] = std

json.dump(d, open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("5 折均值:", mean)
print("5 折标准差:", std)
print("已写回", jp)

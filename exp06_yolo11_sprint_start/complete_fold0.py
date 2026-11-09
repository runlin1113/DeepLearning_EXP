# -*- coding: utf-8 -*-
"""补齐缺失的 Fold 0 评估，并刷新 kfold_metrics.json 的 5 折均值±标准差。"""

# 统一工作目录：以下相对路径均以本文件所在目录为基准，克隆到任意位置可直接运行
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, json, torch
from ultralytics import YOLO

DEVICE = 0 if torch.cuda.is_available() else "cpu"
k = 0
data = f"dataset/kfold/data_fold_{k}.yaml"
name = f"kfold_f{k}"
m = YOLO("yolo11n.pt")
m.train(data=data, epochs=100, imgsz=640, batch=16, device=DEVICE,
        workers=0, patience=30, seed=42, name=name, exist_ok=True, verbose=False)
mm = YOLO(f"runs/detect/{name}/weights/best.pt")
met = mm.val(data=data, device=DEVICE, verbose=False)
res = {"precision": round(float(met.box.mp), 4), "recall": round(float(met.box.mr), 4),
       "map50": round(float(met.box.map50), 4), "map5095": round(float(met.box.map), 4)}

jp = "dataset/kfold/kfold_metrics.json"
d = json.load(open(jp, encoding="utf-8"))
d["folds"][str(k)] = res
valid = [d["folds"][str(i)] for i in range(5) if d["folds"].get(str(i))]
def ms(key):
    xs = [v[key] for v in valid]; m_ = sum(xs) / len(xs)
    var = sum((x - m_) ** 2 for x in xs) / len(xs)
    return round(m_, 4), round(var ** 0.5, 4)
d["kfold_mean"] = {kk: ms(kk)[0] for kk in ["precision", "recall", "map50", "map5095"]}
d["kfold_std"]  = {kk: ms(kk)[1] for kk in ["precision", "recall", "map50", "map5095"]}
json.dump(d, open(jp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("Fold 0:", res)
print("5 折均值±标准差已刷新 ->", d["kfold_mean"], d["kfold_std"])

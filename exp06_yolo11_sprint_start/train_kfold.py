# -*- coding: utf-8 -*-
"""
实验六 · 抗泄漏分组 K 折交叉验证 + 最终部署模型（健壮版）
=========================================================
设计要点（针对 Windows 多进程崩溃的修复）：
  * 加 `if __name__ == "__main__":` 主入口保护，避免 ultralytics 的 spawn
    子进程重复导入并重新执行脚本导致的递归崩溃。
  * 指标不调用 model.val()（其 dataloader 在 Windows 下会 spawn），而是
    直接读取各 run 目录 results.csv 的逐 epoch 指标，取 mAP50-95 最高轮。
  * 断点续跑：若 run 目录已有 best.pt 则跳过训练，只读指标。

运行：venv_xanylabeling/Scripts/python.exe train_kfold.py
"""

# 统一工作目录：以下相对路径均以本文件所在目录为基准，克隆到任意位置可直接运行
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, json, csv
import torch
from ultralytics import YOLO

EPOCHS = 100
N_FOLDS = 5
KFOLD_DIR = "dataset/kfold"
BASE_MODEL = "yolo11n.pt"
SEED = 42
DEVICE = None


def run_dir(name):
    return os.path.join("runs", "detect", name)


def read_best(name):
    """从 results.csv 取 mAP50-95 最高轮的指纹（等价 best.pt 的选点准则）。"""
    p = os.path.join(run_dir(name), "results.csv")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()}
                for r in csv.DictReader(f)]
    if not rows:
        return None

    def g(r, key):
        for k in r:
            if key in k.lower():
                try:
                    return float(r[k])
                except ValueError:
                    pass
        return None

    best = max(rows, key=lambda r: (g(r, "map50-95") or -1))
    return {"precision": round(g(best, "precision") or 0, 4),
            "recall": round(g(best, "recall") or 0, 4),
            "map50": round(g(best, "map50") or 0, 4),
            "map5095": round(g(best, "map50-95") or 0, 4)}


def train_one(name, data_yaml):
    bp = os.path.join(run_dir(name), "weights", "best.pt")
    if os.path.exists(bp):
        print(f"[跳过训练] {name} 已有 best.pt，直接读指标")
        return
    print(f"\n===== 训练 {name} ({data_yaml}) =====")
    m = YOLO(BASE_MODEL)
    m.train(data=data_yaml, epochs=EPOCHS, imgsz=640, batch=16, device=DEVICE,
            workers=0, patience=30, seed=SEED, name=name, exist_ok=True, verbose=False)


def main():
    global DEVICE
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    print(f"[设备] CUDA={torch.cuda.is_available()} device={DEVICE}")

    fold_m = {}
    for k in range(N_FOLDS):
        name = f"kfold_f{k}"
        data = os.path.join(KFOLD_DIR, f"data_fold_{k}.yaml")
        try:
            train_one(name, data)
            fold_m[k] = read_best(name)
            print(f"Fold {k}: {fold_m[k]}")
        except Exception as e:
            print(f"[错误] Fold {k}: {e}")
            fold_m[k] = None

    grouped = allm = None
    for name, data in [("grouped_split", "data_grouped.yaml"),
                       ("start_pose_all52", "data_all.yaml")]:
        try:
            train_one(name, os.path.join(KFOLD_DIR, data))
            r = read_best(name)
            print(f"{name}: {r}")
            if name == "grouped_split":
                grouped = r
            else:
                allm = r
        except Exception as e:
            print(f"[错误] {name}: {e}")

    valid = [fold_m[k] for k in range(N_FOLDS) if fold_m.get(k)]
    def ms(key):
        xs = [v[key] for v in valid]
        m_ = sum(xs) / len(xs)
        var = sum((x - m_) ** 2 for x in xs) / len(xs)
        return round(m_, 4), round(var ** 0.5, 4)

    mean = {k: ms(k)[0] for k in ["precision", "recall", "map50", "map5095"]}
    std = {k: ms(k)[1] for k in ["precision", "recall", "map50", "map5095"]}
    out = {"folds": fold_m, "grouped": grouped, "all52": allm,
           "kfold_mean": mean, "kfold_std": std}
    with open(os.path.join(KFOLD_DIR, "kfold_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n5 折均值±std:", mean, std)
    print("已写入", os.path.join(KFOLD_DIR, "kfold_metrics.json"))


if __name__ == "__main__":
    main()

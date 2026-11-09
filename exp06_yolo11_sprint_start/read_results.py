# -*- coding: utf-8 -*-
"""
不跑 val()（避免 Windows 多进程 spawn 崩溃），直接读各 run 目录的 results.csv
里的逐 epoch 指标，取 mAP50-95 最高的那一轮作为该模型的验证指标。
用于补齐 fold_0 与核对 grouped_split / all52。
"""

# 统一工作目录：以下相对路径均以本文件所在目录为基准，克隆到任意位置可直接运行
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import csv, os

DIRS = ["kfold_f0", "grouped_split", "start_pose_all52", "start_pose_yolo11n"]

def load(path):
    with open(path, encoding="utf-8") as f:
        rows = [{ (k or "").strip(): (v or "").strip() for k, v in r.items() }
                for r in csv.DictReader(f)]
    return rows

def metric_col(row, *names):
    """按列名模糊匹配取指标值。"""
    for k in row:
        kl = k.lower()
        for n in names:
            if n in kl:
                try:
                    return float(row[k])
                except ValueError:
                    pass
    return None

for d in DIRS:
    p = os.path.join("runs", "detect", d, "results.csv")
    if not os.path.exists(p):
        print(f"[{d}] 无 results.csv")
        continue
    rows = load(p)
    if not rows:
        print(f"[{d}] csv 为空")
        continue
    # 以 mAP50-95 最高轮为 best
    def m95(r):
        v = metric_col(r, "map50-95", "mAP50-95")
        return v if v is not None else -1
    best = max(rows, key=m95)
    last = rows[-1]
    def show(tag, r):
        print(f"  {tag:5s}: epoch={r.get('epoch')} "
              f"P={metric_col(r,'precision')} R={metric_col(r,'recall')} "
              f"mAP50={metric_col(r,'map50')} mAP50-95={metric_col(r,'map50-95')}")
    print(f"[{d}] 共 {len(rows)} 轮; csv 列: {[k for k in rows[0] if 'metrics' in k]}")
    show("best", best)
    show("last", last)

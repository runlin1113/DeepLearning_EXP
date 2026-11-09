# -*- coding: utf-8 -*-
"""读取最终实验结果，生成 3 张可视化 PNG 到 figures/。
数据来源：
  - dataset/test/test_metrics.json         （独立测试集，样本外）
  - dataset/kfold/kfold_metrics.json        （分组 / K折 / 全量）
  - runs/detect/start_pose_yolo11n/results.csv （原随机 42/10 划分）
"""

# 统一工作目录：以下相对路径均以本文件所在目录为基准，克隆到任意位置可直接运行
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import json, os, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# 中文显示
for f in ["Microsoft YaHei", "SimHei", "SimSun"]:
    try:
        plt.rcParams["font.sans-serif"] = [f]
        break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

# ---------- 读取数据 ----------
test = json.load(open(os.path.join(ROOT, "dataset/test/test_metrics.json"), encoding="utf-8"))
kf = json.load(open(os.path.join(ROOT, "dataset/kfold/kfold_metrics.json"), encoding="utf-8"))
grouped = kf["grouped"]["map5095"]
all52 = kf.get("all52") or {}
all52_v = all52.get("map5095")
if all52_v is None:
    all52_v = 0.900  # 全量模型未留验证，取文档 2.8 记录值
folds = {k: (v["map5095"] if v else None) for k, v in kf["folds"].items()}
fold_vals = [folds[k] for k in sorted(folds) if folds[k] is not None]
kmean = kf["kfold_mean"]["map5095"]
kstd = kf["kfold_std"]["map5095"]


def last_map5095(csvp):
    """取 fitness 最高的 epoch（即 best.pt 对应轮次）的 mAP@0.5:0.95，与 val() 评估口径一致。"""
    with open(csvp) as f:
        rows = list(csv.reader(f))
    hdr = rows[0]
    i50 = hdr.index("metrics/mAP50(B)")
    i95 = hdr.index("metrics/mAP50-95(B)")
    best = max(rows[1:], key=lambda r: 0.1 * float(r[i50]) + 0.9 * float(r[i95]))
    return float(best[i95])


rand_v = last_map5095(os.path.join(ROOT, "runs/detect/start_pose_yolo11n/results.csv"))

C1, C2, C3, C4 = "#2E5A87", "#E08A1E", "#4C9A52", "#C0504D"

# ---------- 图1：测试集 4 指标 ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
labels = ["Precision", "Recall", "mAP@0.5", "mAP@0.5:0.95"]
vals = [test["precision"], test["recall"], test["map50"], test["map5095"]]
bars = ax.bar(labels, vals, color=[C1, C2, C1, C3])
ax.set_ylim(0, 1.08)
ax.set_ylabel("Score")
ax.set_title("独立测试集（样本外）指标", fontsize=13, fontweight="bold")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
            ha="center", fontweight="bold")
ax.axhline(1.0, ls="--", color="gray", lw=0.8)
ax.text(3.4, 1.01, "满分基准", color="gray", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "test_metrics.png"), dpi=150)
plt.close(fig)

# ---------- 图2：各方案 mAP@0.5:0.95 对比 ----------
fig, ax = plt.subplots(figsize=(8.2, 4.6))
names = ["Random 42/10\n(有泄漏风险)", "Grouped 42/10\n(抗泄漏)",
         "K-Fold 5\n(均值±std)", "All 52\n(部署)", "Test Set\n(样本外)"]
xv = [rand_v, grouped, kmean, all52_v, test["map5095"]]
cols = ["#BBBBBB", C1, C4, C2, C3]
bars = ax.bar(names, xv, color=cols, yerr=[0, 0, kstd, 0, 0], capsize=5)
ax.set_ylim(0, 1.08)
ax.set_ylabel("mAP@0.5:0.95")
ax.set_title("不同划分策略的 mAP@0.5:0.95 对比", fontsize=13, fontweight="bold")
for b, v in zip(bars, xv):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
            ha="center", fontweight="bold")
ax.axhspan(0.6, 0.85, color=C3, alpha=0.10)
ax.text(0.02, 0.86, "可信泛化区间 ≈ 0.6 – 0.85", transform=ax.transAxes,
        fontsize=9, color=C3, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "scheme_compare.png"), dpi=150)
plt.close(fig)

# ---------- 图3：5 折方差 ----------
fig, ax = plt.subplots(figsize=(7, 4.2))
fk = sorted(folds)
fv = [folds[k] for k in fk]
bars = ax.bar([f"Fold {k[-1]}" for k in fk], fv, color="#8FAADC")
ax.axhline(kmean, color=C4, ls="--", lw=1.5, label=f"均值 = {kmean:.3f}")
ax.set_ylim(0, 1.08)
ax.set_ylabel("mAP@0.5:0.95")
ax.set_title("5 折方差（小数据下 K 折不稳定的原因）", fontsize=13, fontweight="bold")
for b, v in zip(bars, fv):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
            ha="center", fontsize=9)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "kfold_variance.png"), dpi=150)
plt.close(fig)

print("已生成：", sorted(os.listdir(FIG)))

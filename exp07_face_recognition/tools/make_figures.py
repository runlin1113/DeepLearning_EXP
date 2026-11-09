"""
生成报告用的示意图（matplotlib 绘制，避免 Word 里 ASCII 框图错位）。
产出 outputs/figures/pipeline.png ：静态识别链路 + 视频动态识别链路

用法： python tools/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
from face_system.utils import setup_mpl_cn  # noqa: E402

setup_mpl_cn()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

C_MAIN = "#2E5C8A"
C_SUB = "#E8F0F8"
C_ACC = "#C0392B"


def box(ax, x, y, w, h, title, sub="", color=C_MAIN, fill=C_SUB, tsize=9.5, ssize=7.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.3, edgecolor=color, facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h * (0.66 if sub else 0.5), title, ha="center", va="center",
            fontsize=tsize, color=color, fontweight="bold", zorder=3)
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=ssize, color="#333333", zorder=3)


def arrow(ax, p1, p2, color=C_MAIN, style="-|>", ls="-", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1))


def draw_static(ax):
    ax.set_title("(a) 静态人脸识别链路", fontsize=11, fontweight="bold", pad=6)
    y, h = 0.44, 0.30
    boxes = [
        ("图像输入", "JPG / PNG", 0.02),
        ("人脸检测", "YuNet / CenterFace\nbbox + 5 关键点", 0.19),
        ("人脸对齐", "evoLVe 相似变换\n→ 112×112", 0.38),
        ("特征提取", "ArcFace 骨干\n→ 512 维 (L2)", 0.57),
        ("余弦比对", "1:1 验证 / 1:N 检索", 0.76),
    ]
    w = 0.165
    for t, s, x in boxes:
        box(ax, x, y, w, h, t, s, color=C_MAIN if t != "人脸对齐" else C_ACC)
    for i in range(len(boxes) - 1):
        arrow(ax, (boxes[i][2] + w, y + h / 2), (boxes[i + 1][2], y + h / 2))

    # 底库
    box(ax, 0.60, 0.06, 0.30, 0.22, "底库 Gallery", "身份 → 均值特征 (npz)",
        color="#4A7A4A", fill="#EAF4EA")
    arrow(ax, (0.75, 0.28), (0.845, 0.44), color="#4A7A4A", style="<|-|>")
    box(ax, 0.19, 0.06, 0.34, 0.22, "阈值 τ 判定", "max cos ≥ τ → 命中，否则“未知”",
        color="#7A5C2E", fill="#FBF3E4")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def draw_video(ax):
    ax.set_title("(b) 视频动态识别链路", fontsize=11, fontweight="bold", pad=6)
    y, h, w = 0.50, 0.26, 0.20
    items = [("视频帧", "25~30 fps", 0.01), ("检测 + 对齐", "逐帧", 0.235),
             ("ArcFace 关键帧", "每 3 帧提一次", 0.46), ("特征 EMA 融合", "α=0.6", 0.685)]
    for t, s, x in items:
        box(ax, x, y, w, h, t, s)
    for i in range(len(items) - 1):
        arrow(ax, (items[i][2] + w, y + h / 2), (items[i + 1][2], y + h / 2))
    box(ax, 0.685, 0.10, 0.20, 0.26, "1:N 检索", "对底库比对", color="#4A7A4A", fill="#EAF4EA")
    arrow(ax, (0.785, 0.50), (0.785, 0.36), color="#4A7A4A")
    box(ax, 0.40, 0.10, 0.24, 0.26, "时序投票 (W=15)", "多数票 + 中位相似度", color="#7A5C2E",
        fill="#FBF3E4")
    arrow(ax, (0.685, 0.23), (0.64, 0.23), color="#7A5C2E")
    box(ax, 0.06, 0.10, 0.30, 0.26, "输出", "轨迹 ID + 身份 + 相似度", color=C_ACC,
        fill="#FBEAE7")
    arrow(ax, (0.40, 0.23), (0.36, 0.23), color=C_ACC)
    ax.text(0.5, 0.44, "轨迹关联：IoU + 匀速预测 + 归一化中心距兜底",
            ha="center", fontsize=7.5, color="#555555")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def main():
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 4.6), dpi=190)
    draw_static(axes[0])
    draw_video(axes[1])
    fig.tight_layout()
    p = OUT / "pipeline.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"→ {p}")


if __name__ == "__main__":
    main()

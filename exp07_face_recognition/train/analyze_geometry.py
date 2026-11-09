"""
Step 7b —— 特征空间几何分析（三种损失的对比量化）
=================================================
训练完三个骨干后，重新加载权重，在同一批数据上计算"特征空间长什么样"：

    intra        : 同类样本对平均余弦相似度（越大=类内越紧）
    inter_mean   : 异类样本对平均余弦相似度（随机对，理论期望≈0，区分度低）
    inter_max    : 类原型的**最难负样本**相似度（max over other classes，越小=类间越开）
    proto_gap    : 1 − max_j cos(c_i, c_j)，类中心之间的最小间隔
    inter_p95    : 全部类对原型相似度的 95 分位

为什么要看 inter_max / proto_gap？
    归一化特征在超球面上近似各向同性，随机取异类对相似度期望就是 0，
    所以"平均类间相似度"几乎没有区分力。真正体现 margin 作用的是**最近邻类中心**，
    也就是检索时最容易混淆的那一对——这才是 margin 直接优化的目标。

用法：
    python train/analyze_geometry.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from face_system.backbone.iresnet import build_backbone  # noqa: E402
from face_system.config import CFG, ensure_dir  # noqa: E402
from face_system.utils import cosine_sim, l2norm, setup_mpl_cn  # noqa: E402
from train.dataset import build_datasets  # noqa: E402

OUT = ensure_dir(CFG.output_dir / "step7")


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    Fs, ys = [], []
    for x, y in loader:
        f = F.normalize(model(x.to(device)), dim=1)
        Fs.append(f.float().cpu().numpy())
        ys.append(y.numpy())
    return np.vstack(Fs), np.concatenate(ys)


def metrics(F_tr, y_tr, F_va, y_va):
    classes = np.unique(y_tr)
    protos = np.stack([l2norm(F_tr[y_tr == c].mean(0)) for c in classes])

    # 类内 / 类间（样本级随机对）
    iu = np.triu_indices(len(y_va), 1)
    S = cosine_sim(F_va, F_va)[iu]
    same = np.equal.outer(y_va, y_va)[iu]
    intra = float(S[same].mean())
    inter_mean = float(S[~same].mean())

    # 类原型级：最难负样本
    P = cosine_sim(protos, protos)
    np.fill_diagonal(P, -2.0)
    inter_max_per_class = P.max(1)
    inter_max = float(inter_max_per_class.mean())
    proto_gap = float((1 - inter_max_per_class).mean())
    inter_p95 = float(np.percentile(P[P > -1.5], 95))

    sims = cosine_sim(F_va, protos)
    pred = classes[np.argmax(sims, 1)]
    rank1 = float((pred == y_va).mean())
    # Rank-5
    order = np.argsort(-sims, 1)[:, :5]
    rank5 = float(np.mean([y_va[i] in classes[order[i]] for i in range(len(y_va))]))

    return {"val_rank1": round(rank1, 4), "val_rank5": round(rank5, 4),
            "intra": round(intra, 4), "inter_mean": round(inter_mean, 4),
            "inter_max": round(inter_max, 4), "proto_gap": round(proto_gap, 4),
            "inter_p95": round(inter_p95, 4),
            "sep_ratio": round((intra - inter_mean) / max(1e-6, 1 - inter_mean), 4)}


def replot_curves():
    """从已保存的 *_metrics.json 重画训练曲线（不必重新训练）"""
    import json as _json
    import matplotlib
    matplotlib.use("Agg")
    setup_mpl_cn()
    import matplotlib.pyplot as plt

    runs = []
    for loss in ("softmax", "cosface", "arcface"):
        f = OUT / f"{loss}_metrics.json"
        if f.exists():
            runs.append(_json.loads(f.read_text(encoding="utf-8")))
    if not runs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.5), dpi=130)
    for r in runs:
        ep = [h["epoch"] for h in r["history"]]
        axes[0].plot(ep, [h["loss"] for h in r["history"]], label=r["loss"])
        ve = [h for h in r["history"] if "val_acc" in h]
        axes[1].plot([h["epoch"] for h in ve], [h["val_acc"] * 100 for h in ve], "o-", ms=3,
                     label=r["loss"])
        axes[2].plot([h["epoch"] for h in ve], [h["intra_sim"] for h in ve], "o-", ms=3,
                     label=r["loss"])
    axes[0].set_title("训练损失"); axes[1].set_title("验证集 Rank-1 (%)")
    axes[2].set_title("验证集类内相似度 intra")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "compare_curves.png"); plt.close(fig)
    print(f"→ {OUT/'compare_curves.png'}（由 metrics.json 重画）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--losses", nargs="+", default=["softmax", "cosface", "arcface"])
    ap.add_argument("--depth", type=int, default=18)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr, va = build_datasets(CFG.lfw_aligned_dir / "train", CFG.lfw_aligned_dir / "valid")
    dl_tr = DataLoader(tr, batch_size=64, shuffle=False)
    dl_va = DataLoader(va, batch_size=64, shuffle=False)

    results, feats = [], {}
    for loss in args.losses:
        ck = OUT / f"{loss}_backbone.pth"
        if not ck.exists():
            print(f"[skip] 缺少 {ck}，先运行 train/train_margin_compare.py")
            continue
        model = build_backbone(args.depth).to(device)
        model.load_state_dict(torch.load(ck, map_location=device)["backbone"])
        F_tr, y_tr = extract(model, dl_tr, device)
        F_va, y_va = extract(model, dl_va, device)
        m = metrics(F_tr, y_tr, F_va, y_va)
        m["loss"] = loss
        results.append(m)
        feats[loss] = (F_tr, y_tr)
        print(f"  {loss:<10} Rank-1={m['val_rank1']*100:5.1f}%  Rank-5={m['val_rank5']*100:5.1f}%  "
              f"intra={m['intra']:.3f}  inter_max={m['inter_max']:.3f}  "
              f"gap={m['proto_gap']:.3f}")

    if not results:
        return
    (OUT / "geometry_compare.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 三联 PCA 图
    import matplotlib
    matplotlib.use("Agg")
    setup_mpl_cn()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(feats), figsize=(4.2 * len(feats), 4.0), dpi=130)
    if len(feats) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab20")
    for ax, (loss, (F_tr, y_tr)) in zip(axes, feats.items()):
        Xc = F_tr - F_tr.mean(0, keepdims=True)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        Z = Xc @ Vt[:2].T
        for i, c in enumerate(np.unique(y_tr)[:20]):
            mm = y_tr == c
            ax.scatter(Z[mm, 0], Z[mm, 1], s=9, color=cmap(i % 20), alpha=0.9)
        r = [x for x in results if x["loss"] == loss][0]
        ax.set_title(f"{loss}\nRank-1 {r['val_rank1']*100:.1f}%  intra {r['intra']:.2f}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("同一骨干 / 同一数据，只换损失函数：特征分布对比（前 20 类）", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "pca_compare.png"); plt.close(fig)

    # ---- 指标柱状对比
    keys = [("val_rank1", "验证集 Rank-1"), ("intra", "类内相似度"),
            ("inter_max", "最难负样本相似度\n(越小越好)"), ("proto_gap", "类中心最小间隔\n(越大越好)")]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.3 * len(keys), 3.1), dpi=130)
    names = [r["loss"] for r in results]
    for ax, (k, title) in zip(axes, keys):
        vals = [r[k] for r in results]
        bars = ax.bar(names, vals, color=["#9aa0a6", "#4a90d9", "#c0392b"][:len(names)])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}" if v < 1.5 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "geometry_bars.png"); plt.close(fig)

    replot_curves()
    print(f"\n→ {OUT/'geometry_compare.json'} / pca_compare.png / geometry_bars.png")


if __name__ == "__main__":
    main()

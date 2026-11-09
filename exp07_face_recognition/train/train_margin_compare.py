"""
Step 7 —— 训练实验：从 Softmax 到 ArcFace（度量学习）
=====================================================
目的：用同一份数据、同一套增强、同一个骨干，只换损失函数，量化"margin 到底带来了什么"。

对照三组：
    1) Softmax  : logits = s·cos(θ)                —— 纯分类，没有显式拉大类间距离
    2) CosFace  : 目标类 logits = s·(cos(θ) − m)    —— 余弦间隔
    3) ArcFace  : 目标类 logits = s·cos(θ + m)      —— 角度间隔（在超球面上直接加角度）

评价指标（不看"分类准确率"这一项，还要看**特征空间的几何性质**）：
    val_acc      : 用训练集算出的类原型做最近邻，验证集 Rank-1（更接近实际检索场景）
    intra_sim    : 同一身份样本对之间的平均余弦相似度（越大越紧）
    inter_sim    : 不同身份样本对之间的平均余弦相似度（越小越开）
    sep_ratio    : (intra − inter) / (1 − inter)，可理解为"分离度"

用法：
    python train/train_margin_compare.py                     # 三种损失全跑（默认 30 epoch）
    python train/train_margin_compare.py --loss arcface -e 60
    python train/train_margin_compare.py --quick             # 5 epoch 快速跑通

注意（重要）：LFW-62 只有 930 张训练图，**从零训**不可能达到工业精度，
本实验要观察的是"三种损失在相同条件下的相对差异"，而不是绝对精度。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from face_system.backbone.iresnet import build_backbone  # noqa: E402
from face_system.config import CFG, ensure_dir  # noqa: E402
from face_system.utils import cosine_sim, l2norm  # noqa: E402
from train.dataset import build_datasets  # noqa: E402
from train.losses import build_head  # noqa: E402

OUT = ensure_dir(CFG.output_dir / "step7")


def set_seed(s: int):
    import random
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    feats, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        f = F.normalize(model(x), dim=1)
        feats.append(f.float().cpu().numpy())
        labels.append(y.numpy())
    return np.vstack(feats), np.concatenate(labels)


def geometry_metrics(F: np.ndarray, y: np.ndarray):
    """intra / inter 相似度与分离度"""
    S = cosine_sim(F, F)
    n = len(y)
    same = np.equal.outer(y, y)
    iu = np.triu_indices(n, 1)
    s, m = S[iu], same[iu]
    intra = float(s[m].mean()) if m.any() else 0.0
    inter = float(s[~m].mean()) if (~m).any() else 0.0
    return {"intra_sim": round(intra, 4), "inter_sim": round(inter, 4),
            "sep_ratio": round((intra - inter) / max(1e-6, 1 - inter), 4)}


def proto_rank1(F_tr, y_tr, F_va, y_va):
    """用训练集类原型做最近邻分类（等价于"注册-检索"流程）"""
    classes = np.unique(y_tr)
    protos = np.stack([l2norm(F_tr[y_tr == c].mean(0)) for c in classes])
    sims = cosine_sim(F_va, protos)
    pred = classes[np.argmax(sims, 1)]
    return float((pred == y_va).mean())


def train_one(loss_mode: str, args, device):
    print(f"\n{'='*58}\n▶ 训练 [{loss_mode}]  epochs={args.epochs}  depth={args.depth}  "
          f"m={args.margin}  s={args.scale}\n{'='*58}")
    set_seed(args.seed)
    tr, va = build_datasets(CFG.lfw_aligned_dir / "train", CFG.lfw_aligned_dir / "valid",
                            image_size=args.image_size, max_per_class=args.max_per_class)
    print(f"  数据: train={len(tr)} 张 / {len(tr.classes)} 类，valid={len(va)} 张")
    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       drop_last=len(tr) > args.batch)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = build_backbone(args.depth, emb_dim=512, input_size=args.image_size).to(device)
    head = build_head(loss_mode, 512, len(tr.classes), args.scale, args.margin).to(device)
    params = list(model.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02)
    ce = nn.CrossEntropyLoss()

    hist = []
    t0 = time.perf_counter()
    for ep in range(1, args.epochs + 1):
        model.train(); head.train()
        tot_loss = tot_acc = 0.0
        nb = 0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            feat = model(x)
            logits = head(feat, y)
            loss = ce(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += float(loss)
            with torch.no_grad():   # 准确率用"无 margin 的余弦 argmax"统计，避免被 margin 压低
                tot_acc += float((head.cosine_logits(feat).argmax(1) == y).float().mean())
            nb += 1
        sched.step()
        rec = {"epoch": ep, "loss": round(tot_loss / max(1, nb), 4),
               "train_acc": round(tot_acc / max(1, nb), 4), "lr": round(opt.param_groups[0]["lr"], 6)}
        if ep % args.eval_every == 0 or ep == args.epochs:
            F_tr, y_tr = extract(model, dl_tr, device)
            F_va, y_va = extract(model, dl_va, device)
            g = geometry_metrics(F_va, y_va)
            rec.update(g)
            rec["val_acc"] = round(proto_rank1(F_tr, y_tr, F_va, y_va), 4)
            print(f"  ep{ep:>3}/{args.epochs}  loss={rec['loss']:.3f}  "
                  f"train_acc={rec['train_acc']*100:5.1f}%  val_rank1={rec['val_acc']*100:5.1f}%  "
                  f"intra={g['intra_sim']:.3f}  inter={g['inter_sim']:.3f}  "
                  f"sep={g['sep_ratio']:.3f}")
        else:
            print(f"  ep{ep:>3}/{args.epochs}  loss={rec['loss']:.3f}  "
                  f"train_acc={rec['train_acc']*100:5.1f}%")
        hist.append(rec)

    dt = time.perf_counter() - t0
    last = [h for h in hist if "val_acc" in h][-1]
    result = {"loss": loss_mode, "epochs": args.epochs, "depth": args.depth,
              "margin": args.margin, "scale": args.scale, "time_s": round(dt, 1),
              "final": last, "history": hist}

    # 保存权重与指标
    torch.save({"backbone": model.state_dict(), "classes": tr.classes,
                "loss": loss_mode, "margin": args.margin},
               OUT / f"{loss_mode}_backbone.pth")
    (OUT / f"{loss_mode}_metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 特征分布 PCA（可视化"margin 把类内压紧、类间推开"）
    try:
        plot_embeddings(F_tr, y_tr, tr.classes, loss_mode)
    except Exception as e:
        print(f"  [warn] PCA 可视化失败: {e}")
    return result


def plot_embeddings(F, y, classes, tag):
    import matplotlib
    matplotlib.use("Agg")
    from face_system.utils import setup_mpl_cn  # noqa: E402
    setup_mpl_cn()
    import matplotlib.pyplot as plt
    # 纯 NumPy 实现 PCA（不额外依赖 sklearn）
    Xc = F - F.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Z = Xc @ Vt[:2].T
    fig, ax = plt.subplots(figsize=(5.4, 5.0), dpi=130)
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(classes[:20]):
        m = y == i
        ax.scatter(Z[m, 0], Z[m, 1], s=10, color=cmap(i % 20), label=c[:12], alpha=0.85)
    ax.set_title(f"训练集特征分布（前 20 类）— {tag}")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=5, ncol=2, loc="best", frameon=False)
    fig.tight_layout(); fig.savefig(OUT / f"pca_{tag}.png"); plt.close(fig)


def plot_compare(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), dpi=130)
    for r in results:
        ep = [h["epoch"] for h in r["history"]]
        axes[0].plot(ep, [h["loss"] for h in r["history"]], label=r["loss"])
        ve = [h for h in r["history"] if "val_acc" in h]
        axes[1].plot([h["epoch"] for h in ve], [h["val_acc"] for h in ve], "o-", ms=3,
                     label=r["loss"])
        axes[2].plot([h["epoch"] for h in ve], [h["sep_ratio"] for h in ve], "o-", ms=3,
                     label=r["loss"])
    axes[0].set_title("训练损失"); axes[1].set_title("验证集 Rank-1")
    axes[2].set_title("特征分离度 sep_ratio")
    for ax in axes:
        ax.set_xlabel("epoch"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "compare_curves.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="all",
                    choices=["all", "softmax", "cosface", "arcface"])
    ap.add_argument("--epochs", "-e", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--depth", type=int, default=18)
    ap.add_argument("--margin", type=float, default=0.35)
    ap.add_argument("--scale", type=float, default=30.0)
    ap.add_argument("--image-size", type=int, default=112)
    ap.add_argument("--max-per-class", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.epochs = 5
        args.eval_every = 1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    modes = ["softmax", "cosface", "arcface"] if args.loss == "all" else [args.loss]
    results = [train_one(m, args, device) for m in modes]

    if len(results) > 1:
        plot_compare(results)
        print("\n" + "=" * 74)
        print(f"{'损失':<10}{'val_Rank1':>10}{'intra':>9}{'inter':>9}{'sep':>9}{'耗时(s)':>10}")
        print("-" * 74)
        for r in results:
            f = r["final"]
            print(f"{r['loss']:<10}{f['val_acc']*100:>9.1f}%{f['intra_sim']:>9.3f}"
                  f"{f['inter_sim']:>9.3f}{f['sep_ratio']:>9.3f}{r['time_s']:>10.1f}")
        print("=" * 74)
        (OUT / "compare_summary.json").write_text(
            json.dumps([{k: r[k] for k in ("loss", "epochs", "margin", "scale", "time_s")}
                        | r["final"] for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8")
        print(f"→ {OUT/'compare_curves.png'}  {OUT/'compare_summary.json'}")
    print("\n✓ Step 7 完成。")


if __name__ == "__main__":
    main()

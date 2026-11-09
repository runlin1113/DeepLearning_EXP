"""
Step 6 —— 实验结果评估（检测 / 对齐 / 1:1 / 1:N）
=================================================
在比 step4 大一个数量级的 LFW 子集上做定量评估（62 身份 / 930 注册图 / 310 探测图），
并额外做两个关键消融：

  A. 对齐 vs 不对齐   同样的检测框，直接 resize 到 112 vs 相似变换对齐
                      → 验证"对齐不是可选项，而是 ArcFace 能否工作的前提"
  B. 检测分辨率       480 / 640 / 1280 三档 max_side 的速度-精度权衡

指标定义：
  1:1  TAR@FAR      给定误接受率下的通过率（越大越好）
       EER          等错误率（FAR=FRR 处，越小越好）
       AUC          ROC 曲线下面积
  1:N  Rank-1/Rank-5 正确身份出现在第 1 / 前 5 位的比例（CMC 曲线）
  检测 detect_rate   图像中至少检出 1 张人脸的比例；det_ms 平均耗时

用法：
    python scripts/step6_evaluate.py
    python scripts/step6_evaluate.py --skip-det-bench   # 跳过三档分辨率测速
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
from face_system.utils import setup_mpl_cn  # noqa: E402
setup_mpl_cn()
import matplotlib.pyplot as plt  # noqa: E402

from face_system import ArcFaceONNX, FaceAligner, Gallery, build_detector  # noqa: E402
from face_system.config import CFG, ensure_dir  # noqa: E402
from face_system.utils import cosine_sim, imread, imwrite, list_images  # noqa: E402

OUT = ensure_dir(CFG.output_dir / "step6")


# ------------------------------------------------------------------ 指标工具
def compute_roc(same_sims, diff_sims):
    """返回 (fpr, tpr, auc, eer, thr_at_eer, thr_at_far...)"""
    scores = np.concatenate([same_sims, diff_sims])
    labels = np.concatenate([np.ones(len(same_sims)), np.zeros(len(diff_sims))])
    order = np.argsort(-scores)
    s, l = scores[order], labels[order]
    tp = np.cumsum(l)
    fp = np.cumsum(1 - l)
    tpr = tp / max(1, l.sum())
    fpr = fp / max(1, (1 - l).sum())
    auc = float(np.trapezoid(tpr, fpr) if hasattr(np, 'trapezoid') else np.trapz(tpr, fpr))
    tpr = np.concatenate([[0], tpr]); fpr = np.concatenate([[0], fpr])
    # EER：FAR 与 FRR 交叉点
    fnr = 1 - tpr
    idx = int(np.argmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    thr_eer = float(s[max(0, idx - 1)]) if idx > 0 else float(s[0])
    # TAR@FAR
    def tar_at(far_target):
        j = np.searchsorted(fpr, far_target)
        j = min(j, len(tpr) - 1)
        return float(tpr[j])
    return {"fpr": fpr, "tpr": tpr, "auc": round(auc, 5), "eer": round(eer, 5),
            "thr_at_eer": round(thr_eer, 4),
            "tar@far1e-1": round(tar_at(0.1), 4),
            "tar@far1e-2": round(tar_at(0.01), 4),
            "tar@far1e-3": round(tar_at(0.001), 4)}


def best_threshold(same_sims, diff_sims):
    """最大化 (TAR + TNR)/2 的阈值"""
    cands = np.unique(np.round(np.concatenate([same_sims, diff_sims]), 3))
    best = (None, -1)
    for t in cands:
        acc = 0.5 * ((same_sims >= t).mean() + (diff_sims < t).mean())
        if acc > best[1]:
            best = (float(t), float(acc))
    return {"thr": best[0], "balanced_acc": round(best[1], 4)}


# ------------------------------------------------------------- 特征抽取工具
def feats_of_dir(root: Path, rec, aligner=None, mode: str = "align", limit: int = 0):
    """mode='align' 用相似变换对齐；mode='resize' 直接用检测框拉伸（消融用）。
    返回 {身份名: (F, 文件名列表)}"""
    # aligner 可能是 FaceAligner，也可能直接是 detector（消融时）
    det = getattr(aligner, "detector", aligner) if aligner is not None else build_detector()
    out = {}
    for sub in sorted([d for d in root.iterdir() if d.is_dir()]):
        imgs = list_images(sub)[:limit or None]
        crops = []
        for p in imgs:
            if mode == "align":
                # 目录里已经是 112x112 对齐图，直接读，避免二次检测导致小图漏检
                c = imread(p)
                if c is None:
                    continue
                crops.append(c)
            else:
                # 消融：用**原始 LFW 图**里的检测框直接 resize 到 112（不做姿态归一化）
                img = imread(CFG.lfw_dir / root.name / sub.name / p.name)
                if img is None:
                    continue
                fs = det.detect(img)
                if not fs:
                    continue
                f0 = max(fs, key=lambda f: f.area)
                crop = f0.crop(img, pad=0.15)
                if crop.size == 0:
                    continue
                crops.append(cv2.resize(crop, (112, 112)))
        if crops:
            out[sub.name] = (rec.get_feat(crops), [p.name for p in imgs])
    return out


def make_pairs(data: dict, seed: int = 0, max_diff: int = 0):
    """生成同人对/异人对的相似度列表"""
    rng = np.random.default_rng(seed)
    names = sorted(data.keys())
    same = []
    for n in names:
        F, _ = data[n]
        if len(F) < 2:
            continue
        S = cosine_sim(F, F)
        iu = np.triu_indices(len(F), 1)
        same.extend(S[iu].tolist())
    diff = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            Fi, _ = data[names[i]]
            Fj, _ = data[names[j]]
            S = cosine_sim(Fi, Fj)
            diff.extend(S.ravel().tolist())
    if max_diff and len(diff) > max_diff:
        idx = rng.choice(len(diff), max_diff, replace=False)
        diff = list(np.array(diff)[idx])
    return np.array(same, np.float32), np.array(diff, np.float32)


# ----------------------------------------------------------------- 各子实验
def eval_detection(aligner, det, bench_resolutions=(480, 640, 1280)):
    print("[1] 检测性能")
    sets = {"本人 15 张": list_images(CFG.raw_face_dir),
            "LFW valid 310 张": list_images(CFG.lfw_dir / "valid", recursive=True)}
    res = {}
    for label, paths in sets.items():
        ok = 0
        ms = []
        nfaces = []
        conf = []
        for p in paths:
            img = imread(p)
            if img is None:
                continue
            t0 = time.perf_counter()
            fs = det.detect(img)
            ms.append((time.perf_counter() - t0) * 1000)
            ok += int(len(fs) > 0)
            nfaces.append(len(fs))
            conf.extend([f.score for f in fs])
        res[label] = {"n": len(paths), "detect_rate": round(ok / max(1, len(paths)), 4),
                      "avg_ms": round(float(np.mean(ms)), 2),
                      "p95_ms": round(float(np.percentile(ms, 95)), 2),
                      "avg_faces": round(float(np.mean(nfaces)), 2),
                      "avg_score": round(float(np.mean(conf)), 4) if conf else 0}
        print(f"    {label}: 检出率 {res[label]['detect_rate']*100:.1f}%  "
              f"平均 {res[label]['avg_ms']:.1f}ms (p95 {res[label]['p95_ms']:.1f}ms)  "
              f"平均人脸数 {res[label]['avg_faces']}")

    # 分辨率-速度权衡（以本人照片为例，原图较大）
    if bench_resolutions:
        print("    分辨率权衡（本人 15 张，不同 max_side）:")
        bench = {}
        paths = list_images(CFG.raw_face_dir)
        for r in bench_resolutions:
            ok, ms = 0, []
            for p in paths:
                img = imread(p)
                if img is None:
                    continue
                t0 = time.perf_counter()
                fs = det.detect(img, max_side=r)
                ms.append((time.perf_counter() - t0) * 1000)
                ok += int(len(fs) > 0)
            bench[str(r)] = {"detect_rate": round(ok / max(1, len(paths)), 4),
                             "avg_ms": round(float(np.mean(ms)), 2)}
            print(f"      max_side={r:<5} 检出率 {bench[str(r)]['detect_rate']*100:5.1f}%  "
                  f"{bench[str(r)]['avg_ms']:6.2f} ms")
        res["resolution_bench"] = bench
    return res


def eval_alignment_ablation(rec, aligner, det):
    print("\n[2] 对齐消融（valid 集 1:1）")
    d_align = feats_of_dir(CFG.lfw_aligned_dir / "valid", rec, aligner, mode="align")
    d_resize = feats_of_dir(CFG.lfw_aligned_dir / "valid", rec, det, mode="resize")
    out = {}
    for tag, d in (("aligned", d_align), ("bbox_resize", d_resize)):
        same, diff = make_pairs(d, max_diff=8000)
        roc = compute_roc(same, diff)
        out[tag] = {"same_mean": round(float(same.mean()), 4), "diff_mean": round(float(diff.mean()), 4),
                    "auc": roc["auc"], "eer": roc["eer"],
                    "tar@far1e-3": roc["tar@far1e-3"],
                    "n_same": int(len(same)), "n_diff": int(len(diff))}
        print(f"    {tag:<12} 同人均值 {same.mean():.3f} / 异人均值 {diff.mean():.3f}  "
              f"AUC={roc['auc']:.4f}  EER={roc['eer']*100:.2f}%  "
              f"TAR@FAR=1e-3={roc['tar@far1e-3']*100:.1f}%")
    return out, (d_align, d_resize)


def eval_verification(rec, aligner, data_valid):
    print("\n[3] 1:1 验证（LFW valid 同/异人对）")
    same, diff = make_pairs(data_valid, max_diff=12000)
    roc = compute_roc(same, diff)
    thr = best_threshold(same, diff)
    print(f"    同人对 {len(same)} 对: 均值 {same.mean():.4f} ± {same.std():.4f}")
    print(f"    异人对 {len(diff)} 对: 均值 {diff.mean():.4f} ± {diff.std():.4f}")
    print(f"    AUC={roc['auc']:.5f}  EER={roc['eer']*100:.2f}%  "
          f"(EER 阈值≈{roc['thr_at_eer']:.3f})")
    print(f"    TAR@FAR=1e-1 {roc['tar@far1e-1']*100:.1f}%  "
          f"@1e-2 {roc['tar@far1e-2']*100:.1f}%  @1e-3 {roc['tar@far1e-3']*100:.1f}%")
    print(f"    最优平衡阈值 τ={thr['thr']:.3f} → 平衡准确率 {thr['balanced_acc']*100:.2f}%")

    # 阈值-精度表
    table = []
    for t in np.arange(0.15, 0.76, 0.05):
        tar = float((same >= t).mean())
        far = float((diff >= t).mean())
        table.append({"thr": round(float(t), 2), "TAR": round(tar, 4), "FAR": round(far, 4),
                      "acc": round((tar + 1 - far) / 2, 4)})
    return {"roc": roc, "best_thr": thr, "table": table,
            "same_mean": round(float(same.mean()), 4),
            "same_std": round(float(same.std()), 4),
            "diff_mean": round(float(diff.mean()), 4),
            "diff_std": round(float(diff.std()), 4),
            "n_same": int(len(same)), "n_diff": int(len(diff))}, (same, diff)


def _degrade_crop(img, kind: str):
    """对已对齐的 112x112 人脸施加成像质量退化（用于质量-精度曲线）"""
    h, w = img.shape[:2]
    if kind == "clean":
        return img
    if kind.startswith("resize"):
        s = int(kind.split("_")[1])
        return cv2.resize(cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA),
                          (w, h), interpolation=cv2.INTER_LINEAR)
    if kind.startswith("jpeg"):
        q = int(kind.split("_")[1])
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR) if ok else img
    if kind == "blur":
        return cv2.GaussianBlur(img, (0, 0), 2.0)
    if kind == "noise":
        rng = np.random.default_rng(0)
        return np.clip(img.astype(np.float32) + rng.normal(0, 14, img.shape), 0, 255).astype(np.uint8)
    if kind == "dark":
        return np.clip(img.astype(np.float32) ** 1.45, 0, 255).astype(np.uint8)
    return img


def eval_degradation(rec, aligner, gallery: Gallery, thr, kinds=None):
    """质量-精度曲线：图越糊/越小，特征越不可靠 → Rank-1 与同人相似度如何下降"""
    kinds = kinds or ["clean", "resize_80", "resize_56", "resize_40", "resize_28",
                      "jpeg_20", "jpeg_5", "blur", "noise", "dark"]
    print("\n[5] 质量鲁棒性（对探测图施加退化）")
    valid_root = CFG.lfw_aligned_dir / "valid"
    # 预先读出全部探测 crop（干净版）
    probes, labels = [], []
    for name in gallery.names:
        for p in list_images(valid_root / name)[:5]:
            c = imread(p)
            if c is not None:
                probes.append(c)
                labels.append(name)
    print(f"    探测样本 {len(probes)} 张")
    out = []
    for kind in kinds:
        F = rec.get_feat([_degrade_crop(c, kind) for c in probes])
        r1 = 0
        sims_true = []
        for f, nm in zip(F, labels):
            pred, sim, res = gallery.identify(f, topk=1)
            r1 += int(pred == nm)
            sims_true.append(float(cosine_sim(f, gallery.id_feats[gallery.names.index(nm)])[0, 0]))
        out.append({"kind": kind, "rank1": round(r1 / len(labels), 4),
                    "mean_sim_true": round(float(np.mean(sims_true)), 4),
                    "accept_rate": round(float(np.mean(np.array(sims_true) >= thr)), 4)})
        print(f"    {kind:<10} Rank-1={out[-1]['rank1']*100:5.1f}%  "
              f"同人相似度 {out[-1]['mean_sim_true']:.3f}  "
              f"过阈值率 {out[-1]['accept_rate']*100:5.1f}%")
    return out


def eval_identification(rec, gallery: Gallery, data_valid):
    print("\n[4] 1:N 检索（62 身份底库，valid 作探测）")
    ranks = []
    sims_true = []
    wrong = []
    for name in gallery.names:
        if name not in data_valid:
            continue
        F, fnames = data_valid[name]
        for i, f in enumerate(F):
            pred, sim, res = gallery.identify(f, topk=len(gallery.names))
            ranking = [n for n, _ in res]
            try:
                r = ranking.index(name) + 1
            except ValueError:
                r = len(ranking) + 1
            ranks.append(r)
            sims_true.append(float(cosine_sim(f, gallery.id_feats[gallery.names.index(name)])[0, 0]))
            if r != 1:
                wrong.append({"truth": name, "pred": ranking[0], "rank": r,
                              "sim_pred": round(res[0][1], 4)})
    ranks = np.array(ranks)
    n = len(ranks)
    cmc = [float((ranks <= k).mean()) for k in range(1, 11)]
    print(f"    探测 {n} 张: Rank-1={cmc[0]*100:.2f}%  Rank-5={cmc[4]*100:.2f}%  "
          f"Rank-10={cmc[9]*100:.2f}%")
    print(f"    真实身份相似度均值 {np.mean(sims_true):.4f}  错误 {len(wrong)} 张")
    return {"n_probe": n, "rank1": round(cmc[0], 4), "rank5": round(cmc[4], 4),
            "cmc": [round(c, 4) for c in cmc],
            "mean_sim_true": round(float(np.mean(sims_true)), 4),
            "errors": wrong[:15]}


# ------------------------------------------------------------------- 绘图
def plot_all(verif, same, diff, ident, align_ab, tag=""):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.7), dpi=130)

    ax = axes[0]
    ax.plot(verif["roc"]["fpr"], verif["roc"]["tpr"], lw=1.8, color="#c0392b",
            label=f"AUC={verif['roc']['auc']:.4f}\nEER={verif['roc']['eer']*100:.2f}%")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FAR"); ax.set_ylabel("TAR"); ax.set_title("1:1 ROC（LFW-62）")
    ax.legend(fontsize=8); ax.grid(alpha=.3)

    ax = axes[1]
    ax.hist(same, bins=40, alpha=0.75, color="#c0392b", label=f"同人 {len(same)} 对")
    ax.hist(diff, bins=40, alpha=0.6, color="#2e86ab", label=f"异人 {len(diff)} 对")
    ax.axvline(verif["best_thr"]["thr"], color="k", ls="--", lw=1,
               label=f"τ={verif['best_thr']['thr']:.2f}")
    ax.set_xlabel("余弦相似度"); ax.set_ylabel("对数（log）"); ax.set_title("相似度分布")
    ax.set_yscale("log")
    ax.legend(fontsize=7); ax.grid(alpha=.3, which="both")

    ax = axes[2]
    ks = np.arange(1, len(ident["cmc"]) + 1)
    ax.plot(ks, np.array(ident["cmc"]) * 100, "o-", ms=4, color="#2e7d32",
            label=f"对齐   Rank-1 {ident['cmc'][0]*100:.1f}%")
    ax.set_xlabel("Rank K"); ax.set_ylabel("识别率 %"); ax.set_title("1:N CMC")
    ax.set_ylim(max(0, ident["cmc"][0] * 100 - 8), 100.5)
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / f"eval_summary{tag}.png"); plt.close(fig)

    # 对齐消融对比图
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=130)
    keys = ["aligned", "bbox_resize"]
    vals = [align_ab[k]["auc"] for k in keys]
    bars = ax.bar(["相似变换对齐", "检测框直接拉伸"], vals,
                  color=["#2e7d32", "#b0b0b0"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.4f}",
                ha="center", fontsize=9)
    ax.set_ylim(min(vals) - 0.05, 1.0)
    ax.set_ylabel("1:1 AUC")
    ax.set_title("对齐消融：AUC 对比")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "ablation_align.png"); plt.close(fig)


def plot_degradation(deg):
    figs = OUT / "quality_curve.png"
    fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=130)
    xs = np.arange(len(deg))
    ax2 = ax.twinx()
    ax.bar(xs, [d["rank1"] * 100 for d in deg], color="#4a90d9", alpha=.75, label="Rank-1")
    ax2.plot(xs, [d["mean_sim_true"] for d in deg], "o-", color="#c0392b", label="同人相似度")
    ax.set_xticks(xs, [d["kind"] for d in deg], rotation=30, fontsize=7)
    ax.set_ylabel("Rank-1 (%)"); ax2.set_ylabel("同人平均余弦相似度")
    ax.set_ylim(0, 108); ax2.set_ylim(0, 1.0)
    ax.set_title("成像质量 vs 识别性能")
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(figs); plt.close(fig)


def vis_alignment_ablation(rec, aligner, det, n_ids: int = 6):
    """左边"检测框直接拉伸"，右边"相似变换对齐"，直观看出姿态归一化的作用"""
    from face_system.utils import draw_text_cn
    valid_root = CFG.lfw_aligned_dir / "valid"
    names = sorted([d.name for d in valid_root.iterdir() if d.is_dir()])[:n_ids]
    rows = []
    for name in names:
        p = list_images(valid_root / name)[:1]
        if not p:
            continue
        raw = imread(CFG.lfw_dir / "valid" / name / p[0].name)
        if raw is None:
            continue
        fs = det.detect(raw)
        if not fs:
            continue
        f0 = max(fs, key=lambda f: f.area)
        crop = f0.crop(raw, pad=0.15)
        naive = cv2.resize(crop, (112, 112))
        aligned = imread(p[0])
        rows.append((naive, aligned, name))
    if not rows:
        return
    # 每个小块 = 左(拉伸) | 右(对齐)，排成 3 列 × 2 行的网格，便于放进报告
    from face_system.utils import make_grid
    gap = np.full((112, 4, 3), 255, np.uint8)
    tiles = [np.hstack([nv, gap, al]) for nv, al, _ in rows]
    grid = make_grid(tiles, cols=3, pad=8, bg=(255, 255, 255))
    head = np.full((30, grid.shape[1], 3), 255, np.uint8)
    head = draw_text_cn(head, "每个小块：左 = 检测框直接拉伸    右 = 相似变换对齐",
                        (6, 7), color=(20, 20, 20), size=14, bg=False)
    imwrite(OUT / "ablation_align_vis.jpg", np.vstack([head, grid]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--skip-det-bench", action="store_true")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    print("=" * 62)
    print(" Step 6  定量评估")
    print("=" * 62 + "\n")

    det = build_detector(prefer_gpu=args.gpu)
    aligner = FaceAligner(det)
    rec = ArcFaceONNX(CFG.rec.model, prefer_gpu=args.gpu)
    gallery = Gallery.load(CFG.output_dir / "gallery" / "gallery_lfw62.npz").finalize()

    det_res = eval_detection(aligner, det, None if args.skip_det_bench else (480, 640, 1280))
    align_ab, (d_align, d_resize) = eval_alignment_ablation(rec, aligner, det)
    verif, (same, diff) = eval_verification(rec, aligner, d_align)
    ident = eval_identification(rec, gallery, d_align)
    degrade = eval_degradation(rec, aligner, gallery, CFG.rec.match_threshold)

    plot_all(verif, same, diff, ident, align_ab, args.tag)
    plot_degradation(degrade)
    vis_alignment_ablation(rec, aligner, det)

    res = {"detector": str(det), "recognizer": str(rec),
           "threshold_cfg": CFG.rec.match_threshold,
           "detection": det_res, "alignment_ablation": align_ab,
           "verification_1to1": {k: v for k, v in verif.items() if k != "roc"}
           | {"auc": verif["roc"]["auc"], "eer": verif["roc"]["eer"],
              "tar@far1e-1": verif["roc"]["tar@far1e-1"],
              "tar@far1e-2": verif["roc"]["tar@far1e-2"],
              "tar@far1e-3": verif["roc"]["tar@far1e-3"]},
           "identification_1toN": ident,
           "quality_robustness": degrade}
    (OUT / f"eval_results{args.tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {OUT/f'eval_results{args.tag}.json'}")
    print(f"图表: {OUT/f'eval_summary{args.tag}.png'} / {OUT/'ablation_align.png'}")
    print("\n✓ Step 6 完成。")


if __name__ == "__main__":
    main()

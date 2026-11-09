"""
Step 4 —— 静态人脸识别测试（1:1 验证 + 1:N 检索）
=================================================
1:1 (verification)：给两张脸，判断是不是同一个人 → 二分类 + 阈值
1:N (identification)：给一张脸，在 M 个注册身份里找是谁 → 排序问题（Rank-1/Rank-5）

测试协议（本人数据只有 15 张，必须严格划分）：
    注册集 = 前 8 张（step3 用）
    探测集 = 后 7 张（本脚本用）
    → 同人对 = 7×8 = 56 对；异人对 = 7 张 × 12 个 LFW 身份
    陌生人 = LFW 中**未注册**的 50 个身份（各 5 张）→ 用于测误接受率 FAR

用法：
    python scripts/step4_static_test.py                     # 跑全部测试
    python scripts/step4_static_test.py --img1 a.jpg --img2 b.jpg   # 只做一张 1:1
    python scripts/step4_static_test.py --thr 0.4
"""
from __future__ import annotations

import argparse
import json
import sys
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
from face_system.utils import (cosine_sim, draw_face_box, fit_font_size,  # noqa: E402
                               imread, imwrite, list_images, make_grid,
                               put_text_scale)

OUT = ensure_dir(CFG.output_dir / "step4")
GAL_DIR = CFG.output_dir / "gallery"


# ------------------------------------------------------------------ 基础工具
def crops_of_paths(paths, aligner, rec):
    """原图 → 最大人脸 → 对齐 → 特征"""
    crops, feats, names = [], [], []
    for p in paths:
        img = imread(p)
        if img is None:
            continue
        afs = aligner.run(img)
        if not afs:
            continue
        a = max(afs, key=lambda x: x.face.area)
        crops.append(a.crop)
        names.append(Path(p).name)
    if crops:
        feats = rec.get_feat(crops)
    return crops, feats, names


def pane(img, text, size=(128, 128), fs=12):
    """统一尺寸的贴片：上图下文；字号按实测文字宽度自适应，长名字也不裁字。"""
    img = cv2.resize(img, size)
    canvas = np.full((size[1] + 24, size[0], 3), 30, np.uint8)
    canvas[:size[1]] = img
    fs = fit_font_size(text, size[0] - 4, fs)
    return put_text_scale(canvas, text, (2, size[1] + 3), color=(235, 235, 235), size=fs)


def gallery_crops(gallery: Gallery, aligned_root: Path, names, k=1):
    """取底库身份的若干张对齐图（仅用于可视化）"""
    out = {}
    for n in names:
        d = aligned_root / n
        imgs = list_images(d)[:k] if d.exists() else []
        out[n] = [imread(p) for p in imgs if imread(p) is not None]
    return out


# ------------------------------------------------------------------ 1:1 验证
def test_1to1(aligner, rec, gallery: Gallery, split, thr):
    print("[1] 1:1 人脸验证")
    reg_paths = [CFG.raw_face_dir / n for n in split["register"]]
    prb_paths = [CFG.raw_face_dir / n for n in split["probe"]]
    _, F_reg, _ = crops_of_paths(reg_paths, aligner, rec)
    prb_crops, F_prb, prb_names = crops_of_paths(prb_paths, aligner, rec)
    if F_prb is None or F_reg is None:
        raise SystemExit("本人照片特征提取失败")

    # --- 同人对
    S_same = cosine_sim(F_prb, F_reg)
    same = S_same.ravel()
    # --- 异人对：探测图 vs 其他注册身份（各身份的均值特征）
    other_idx = [i for i, n in enumerate(gallery.names) if n != "self"]
    S_diff = cosine_sim(F_prb, gallery.id_feats[other_idx])
    diff = S_diff.ravel()

    def stat(x):
        return {"n": int(x.size), "mean": round(float(x.mean()), 4),
                "std": round(float(x.std()), 4), "min": round(float(x.min()), 4),
                "max": round(float(x.max()), 4),
                "p5": round(float(np.percentile(x, 5)), 4),
                "p95": round(float(np.percentile(x, 95)), 4)}

    acc_same = float((same >= thr).mean())
    acc_diff_reject = float((diff < thr).mean())
    print(f"    同人对 ({same.size} 对): 均值 {same.mean():.4f}  最小 {same.min():.4f}  "
          f"P5 {np.percentile(same,5):.4f}")
    print(f"    异人对 ({diff.size} 对): 均值 {diff.mean():.4f}  最大 {diff.max():.4f}  "
          f"P95 {np.percentile(diff,95):.4f}")
    print(f"    τ={thr:.2f} → 同人通过率 TAR={acc_same*100:.1f}%，"
          f"异人正确拒绝率 TNR={acc_diff_reject*100:.1f}%")

    # --- 阈值扫描（等错误率 EER 在 step6 用大量样本算，这里给个小规模扫描）
    sweep = []
    for t in np.arange(0.10, 0.81, 0.05):
        tar = float((same >= t).mean())
        far = float((diff >= t).mean())
        sweep.append({"thr": round(float(t), 2), "TAR": round(tar, 4), "FAR": round(far, 4),
                      "acc": round((tar + (1 - far)) / 2, 4)})
    best = max(sweep, key=lambda r: r["acc"])
    print(f"    阈值扫描最优: τ={best['thr']}  平衡准确率={best['acc']*100:.1f}% "
          f"(TAR {best['TAR']*100:.1f}% / FAR {best['FAR']*100:.1f}%)")

    # --- 可视化：每张探测图 + 与注册集均值特征的相似度条
    panels = []
    for i, (c, n) in enumerate(zip(prb_crops, prb_names)):
        sim = float(cosine_sim(F_prb[i], gallery.id_feats[gallery.names.index("self")])[0, 0])
        color = (0, 200, 0) if sim >= thr else (0, 0, 220)
        tile = pane(c, f"{sim:.3f}", size=(112, 112))
        cv2.rectangle(tile, (0, 0), (111, 111), color, 2)
        panels.append(tile)
    imwrite(OUT / "1to1_probes.jpg", make_grid(panels, cols=7, titles=[
        f"probe{i+1}" for i in range(len(panels))]))

    # --- 相似度分布直方图
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=130)
    ax.hist(same, bins=20, alpha=0.75, label=f"同人对 (n={same.size})", color="#e4572e")
    ax.hist(diff, bins=20, alpha=0.75, label=f"异人对 (n={diff.size})", color="#2e86ab")
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"阈值 τ={thr:.2f}")
    ax.set_xlabel("余弦相似度"); ax.set_ylabel("样本数")
    ax.set_title("1:1 相似度分布（本人）")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / "1to1_hist.png"); plt.close(fig)

    return {
        "same": stat(same), "diff": stat(diff),
        "thr": thr, "TAR": round(acc_same, 4), "TNR": round(acc_diff_reject, 4),
        "sweep": sweep, "best": best,
    }


# ------------------------------------------------------------------ 1:N 检索
def test_1toN(aligner, rec, gallery: Gallery, thr, topk=5):
    print("\n[2] 1:N 人脸检索")
    aligned_valid = CFG.lfw_aligned_dir / "valid"
    gal_names = [n for n in gallery.names]
    known = [n for n in gal_names if n != "self"]

    # --- (a) 本人 7 张探测图
    split = json.loads((GAL_DIR / "self_split.json").read_text(encoding="utf-8"))
    split = {"register": [n for n in split["register"]], "probe": split["probe"]}
    prb_crops, F_prb, _ = crops_of_paths([CFG.raw_face_dir / n for n in split["probe"]],
                                         aligner, rec)
    hit = 0
    lines = []
    for i in range(len(F_prb)):
        name, sim, res = gallery.identify(F_prb[i], threshold=thr, topk=topk)
        hit += int(name == "self")
        lines.append(f"    probe{i+1}: → {name:<22} sim={sim:.4f}  "
                     f"top3={[(n, round(s,3)) for n, s in res[:3]]}")
    print(f"    本人探测 {len(F_prb)} 张，Rank-1 = {hit}/{len(F_prb)} = "
          f"{hit/max(1,len(F_prb))*100:.1f}%")
    for l in lines[:4]:
        print(l)

    # --- (b) 已注册 LFW 身份的 valid 图（跨域、跨姿态，更难）
    r1 = r5 = tot = 0
    bad = []
    for name in known:
        imgs = list_images(aligned_valid / name)
        crops = [imread(p) for p in imgs]
        crops = [c for c in crops if c is not None]
        if not crops:
            continue
        F = rec.get_feat(crops)
        for j, f in enumerate(F):
            pred, sim, res = gallery.identify(f, threshold=thr, topk=topk)
            ranking = [n for n, _ in res]
            tot += 1
            r1 += int(pred == name)
            r5 += int(name in ranking)
            if pred != name:
                bad.append({"truth": name, "pred": pred, "sim": round(sim, 4),
                            "top1_sim_true": round(float(cosine_sim(f, gallery.id_feats[gallery.names.index(name)])[0, 0]), 4)})
    print(f"    已注册身份回溯 {tot} 张: Rank-1={r1/tot*100:.1f}%  "
          f"Rank-{topk}={r5/tot*100:.1f}%  误识 {len(bad)} 张")

    # --- (c) 陌生人拒识（未注册身份）
    all_train_ids = {d.name for d in (CFG.lfw_aligned_dir / "train").iterdir() if d.is_dir()}
    unknown_ids = sorted(all_train_ids - set(known))
    fa = tot_u = 0
    fake_hits = []
    for name in unknown_ids[:30]:
        imgs = list_images(CFG.lfw_aligned_dir / "valid" / name)[:3]
        crops = [imread(p) for p in imgs]
        crops = [c for c in crops if c is not None]
        if not crops:
            continue
        F = rec.get_feat(crops)
        for f in F:
            pred, sim, res = gallery.identify(f, threshold=thr, topk=1)
            tot_u += 1
            if pred != "未知":
                fa += 1
                fake_hits.append({"truth": name, "pred": pred, "sim": round(sim, 4)})
    far = fa / max(1, tot_u)
    print(f"    陌生人（{len(unknown_ids)} 个未注册身份）{tot_u} 张: "
          f"误接受 {fa} 张，FAR={far*100:.2f}%（τ={thr:.2f}）")

    # --- 可视化：探测图 + top3 底库人脸
    vis_panels = []
    src_crops = gallery_crops(gallery, CFG.lfw_aligned_dir / "train", known, k=1)
    # 补上"本人"的底库缩略图（对齐第一张注册图）
    if "self" in gallery.names:
        reg = json.loads((GAL_DIR / "self_split.json").read_text(encoding="utf-8"))["register"]
        if reg:
            afs = aligner.run(imread(CFG.raw_face_dir / reg[0]))
            if afs:
                src_crops["self"] = [max(afs, key=lambda a: a.face.area).crop]
    sz = (120, 120)
    for i in range(min(4, len(F_prb))):
        name, sim, res = gallery.identify(F_prb[i], threshold=thr, topk=3)
        row = [pane(prb_crops[i], f"probe {i+1}", sz)]
        for rname, rsim in res:
            c = (src_crops.get(rname) or [np.full((120, 120, 3), 30, np.uint8)])[0]
            label = rname if len(rname) <= 16 else rname[:15] + "…"
            row.append(pane(c, f"{label} {rsim:.2f}", sz))
        vis_panels.append(np.hstack(row))
    if vis_panels:
        imwrite(OUT / "1toN_top3.jpg", np.vstack(vis_panels))

    # --- 相似度矩阵热图（探测 × 底库身份）
    M = cosine_sim(F_prb, gallery.id_feats)
    fig, ax = plt.subplots(figsize=(7.6, 2.9), dpi=130)
    im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(M.shape[0]), [f"probe{i+1}" for i in range(M.shape[0])])
    ax.set_xticks(range(M.shape[1]), [n[:11] for n in gallery.names], rotation=60, fontsize=7)
    ax.set_title("本人探测图 × 13 个注册身份  余弦相似度矩阵")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout(); fig.savefig(OUT / "1toN_heatmap.png"); plt.close(fig)

    return {
        "self_rank1": hit / max(1, len(F_prb)),
        "known_rank1": r1 / max(1, tot), "known_rankk": r5 / max(1, tot),
        "known_n": tot, "known_errors": bad[:20],
        "unknown_n": tot_u, "unknown_fa": fa, "FAR": far,
        "n_unknown_ids": len(unknown_ids),
    }


# ------------------------------------------------------------------ 单对 1:1
def single_pair(img1, img2, aligner, rec, thr):
    print("[0] 单张 1:1 比对")
    out = []
    for p in (img1, img2):
        img = imread(p)
        if img is None:
            raise SystemExit(f"读不到图片: {p}")
        afs = aligner.run(img)
        if not afs:
            raise SystemExit(f"图中没有人脸: {p}")
        a = max(afs, key=lambda x: x.face.area)
        vis = draw_face_box(img.copy(), a.face.bbox, a.face.landmarks)
        imwrite(OUT / f"pair_{Path(p).stem[:20]}_det.jpg", vis)
        out.append(a.crop)
    F = rec.get_feat(out)
    sim = float(cosine_sim(F[0], F[1])[0, 0])
    ok = sim >= thr
    print(f"    {Path(img1).name}  vs  {Path(img2).name}")
    print(f"    余弦相似度 = {sim:.4f}   阈值 {thr:.2f} → "
          f"{'同一人 ✓' if ok else '不同人 ✗'}")
    vis = np.hstack([pane(out[0], Path(img1).stem[:16], (160, 160)),
                     pane(out[1], Path(img2).stem[:16], (160, 160))])
    vis = put_text_scale(vis, f"cos={sim:.3f} -> {'SAME' if ok else 'DIFF'}",
                         (2, 2), color=(0, 255, 0) if ok else (0, 0, 255), size=18)
    imwrite(OUT / "1to1_pair.jpg", vis)
    return {"sim": sim, "same": ok, "thr": thr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default=str(GAL_DIR / "gallery_demo.npz"))
    ap.add_argument("--img1"); ap.add_argument("--img2")
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    print("=" * 62)
    print(" Step 4  静态识别测试")
    print("=" * 62 + "\n")

    det = build_detector(prefer_gpu=args.gpu)
    aligner = FaceAligner(det, crop_size=CFG.align.crop_size)
    rec = ArcFaceONNX(CFG.rec.model, prefer_gpu=args.gpu)
    gallery = Gallery.load(args.gallery).finalize()
    thr = args.thr if args.thr is not None else gallery.threshold
    print(f"底库 {gallery.size} 个身份，阈值 τ={thr:.2f}，"
          f"特征 {rec.provider}\n")

    if args.img1 and args.img2:
        res = single_pair(args.img1, args.img2, aligner, rec, thr)
        (OUT / "pair_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
        return

    split = json.loads((GAL_DIR / "self_split.json").read_text(encoding="utf-8"))
    res = {
        "config": {"gallery": str(args.gallery), "threshold": thr,
                   "n_identities": gallery.size},
        "1to1": test_1to1(aligner, rec, gallery, split, thr),
        "1toN": test_1toN(aligner, rec, gallery, thr),
    }
    (OUT / "results_static.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"\n结果已保存: {OUT/'results_static.json'}")
    print("✓ Step 4 完成。下一步： python scripts/step5_video_demo.py --make-test-video")


if __name__ == "__main__":
    main()

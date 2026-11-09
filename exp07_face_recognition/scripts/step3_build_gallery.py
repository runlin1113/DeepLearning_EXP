"""
Step 3 —— 构建人脸底库（Gallery）
=================================
底库 = "已注册身份 → 特征"。本实验建两个库：

    outputs/gallery/gallery_lfw62.npz   LFW 62 个身份的 train 图（1:N 评估用）
    outputs/gallery/gallery_demo.npz    本人(注册用前 8 张) + 12 个 LFW 干扰身份（演示用）

要点：
  1) 一个身份注册多张图 → 特征取**均值后再 L2 归一化**（等价于单位球上的质心），
     比只用单张更稳；同时保留逐脸特征，支持"逐脸 top-k 均分"检索策略；
  2) 注册集与探测集**必须分开**（本人 15 张里 8 张注册、7 张探测），
     否则相似度虚高，实验无效；
  3) 特征全部 L2 归一化后存 npz，余弦相似度 = 内积，检索就是一次矩阵乘。

用法：
    python scripts/step3_build_gallery.py
    python scripts/step3_build_gallery.py --n-reg 8 --n-lfw-demo 12
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from face_system import ArcFaceONNX, FaceAligner, build_detector, Gallery  # noqa: E402
from face_system.config import CFG, ensure_dir  # noqa: E402
from face_system.utils import imread, list_images  # noqa: E402

GAL_DIR = ensure_dir(CFG.output_dir / "gallery")
SPLIT_JSON = GAL_DIR / "self_split.json"


def build_self(gallery: Gallery, rec, aligner, n_reg: int):
    """本人身份：前 n_reg 张注册，其余留作探测集。"""
    paths = list_images(CFG.raw_face_dir)
    if not paths:
        raise SystemExit(f"没找到本人照片: {CFG.raw_face_dir}")
    crops, used = [], []
    for p in paths:
        img = imread(p)
        if img is None:
            continue
        afs = aligner.run(img)
        if not afs:
            print(f"    [warn] 未检出人脸: {p.name}")
            continue
        crops.append(max(afs, key=lambda a: a.face.area).crop)
        used.append(p.name)

    reg = crops[:n_reg]
    probe = crops[n_reg:]
    feats = rec.get_feat(reg)
    gallery.add_identity("self", feats,
                         {"n_images": len(reg), "source": "data/images/self"})
    split = {"register": used[:n_reg], "probe": used[n_reg:]}
    SPLIT_JSON.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    本人: 注册 {len(reg)} 张 / 探测 {len(probe)} 张 → {SPLIT_JSON.name}")
    return split


def build_lfw(gallery: Gallery, rec, aligned_root: Path, ids=None, per_id: int = 0):
    """从已对齐的 LFW 目录注册，返回注册的身份名列表。"""
    dirs = sorted([p for p in aligned_root.iterdir() if p.is_dir()])
    if ids is not None:
        dirs = [d for d in dirs if d.name in ids]
    names = []
    for d in dirs:
        imgs = list_images(d)[:per_id or None]
        crops = [imread(p) for p in imgs]
        crops = [c for c in crops if c is not None]
        if not crops:
            continue
        gallery.add_identity(d.name, rec.get_feat(crops),
                             {"n_images": len(crops), "source": f"lfw20_aligned/{aligned_root.name}"})
        names.append(d.name)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reg", type=int, default=8, help="本人注册用张数")
    ap.add_argument("--n-lfw-demo", type=int, default=12, help="演示库里的 LFW 干扰身份数")
    ap.add_argument("--per-id", type=int, default=0, help="每个 LFW 身份注册几张（0=全部）")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    print("=" * 62)
    print(" Step 3  构建底库")
    print("=" * 62 + "\n")

    det = build_detector(prefer_gpu=args.gpu)
    aligner = FaceAligner(det, crop_size=CFG.align.crop_size)
    rec = ArcFaceONNX(CFG.rec.model, prefer_gpu=args.gpu)
    aligned_train = CFG.lfw_aligned_dir / "train"
    if not aligned_train.exists():
        raise SystemExit("请先运行 scripts/step2_detect_align.py 生成对齐后的 LFW 数据集")

    # ---------- 1) LFW 全量底库（62 身份，用于 1:N 评估）
    print("[1] gallery_lfw62.npz")
    g1 = Gallery()
    names = build_lfw(g1, rec, aligned_train, per_id=args.per_id)
    g1.finalize()
    g1.save(GAL_DIR / "gallery_lfw62.npz")
    print(f"    {g1.size} 个身份，{sum(len(f) for f in g1.face_feats)} 张人脸特征 "
          f"→ {(GAL_DIR/'gallery_lfw62.npz').name}")

    # ---------- 2) 演示底库（本人 + 若干干扰身份）
    print("\n[2] gallery_demo.npz")
    g2 = Gallery()
    split = build_self(g2, rec, aligner, args.n_reg)
    demo_ids = set(names[:args.n_lfw_demo])
    build_lfw(g2, rec, aligned_train, ids=demo_ids, per_id=args.per_id)
    g2.finalize()
    g2.save(GAL_DIR / "gallery_demo.npz")
    print(f"    {g2.size} 个身份 → {(GAL_DIR/'gallery_demo.npz').name}")
    print("\n" + g2.summary()[:1200])

    # ---------- 3) 一致性自检
    print("\n[3] 底库自检（用注册图自身检索，Rank-1 应=100%）")
    hit = 0
    total = 0
    for i, name in enumerate(g2.names):
        for f in g2.face_feats[i][:2]:
            total += 1
            hit += int(g2.identify(f, topk=1)[0] == name)
    print(f"    自检索命中 {hit}/{total} = {hit/max(1,total)*100:.1f}%")
    print("\n✓ Step 3 完成。下一步： python scripts/step4_static_test.py")


if __name__ == "__main__":
    main()

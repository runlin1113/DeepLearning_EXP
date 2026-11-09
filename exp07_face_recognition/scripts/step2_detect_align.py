"""
Step 2 —— 人脸检测 + 对齐（本实验的核心链路第一步）
===================================================
对本人 15 张原始照片做检测/对齐可视化；把 LFW 公开小样本库批量对齐成 112x112，
供后面训练与评估使用。

用法：
    python scripts/step2_detect_align.py                 # 本人照片可视化 + LFW 批量对齐
    python scripts/step2_detect_align.py --no-lfw        # 只处理本人照片
    python scripts/step2_detect_align.py --detector centerface

产出：
    outputs/step2/detection_grid.jpg    检测框 + 5 关键点
    outputs/step2/alignment_grid.jpg    112x112 对齐结果
    outputs/step2/stats.json            检测率/耗时/双眼水平差/左右互换统计
    data/lfw20_aligned/{train,valid}/<身份>/*.jpg
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

from face_system import build_detector, FaceAligner  # noqa: E402
from face_system.align.align_pipeline import align_folder  # noqa: E402
from face_system.config import CFG, ensure_dir  # noqa: E402
from face_system.utils import (draw_face_box, imread, imwrite, list_images,  # noqa: E402
                               make_grid, Timer)


def process_personal(aligner, out_dir: Path, cols: int = 5):
    """本人照片：逐张检测 + 对齐 + 拼图 + 统计"""
    paths = list_images(CFG.raw_face_dir)
    if not paths:
        print(f"[skip] {CFG.raw_face_dir} 下没有图片")
        return None

    det_tiles, crop_tiles, titles = [], [], []
    rec = []
    for p in paths:
        img = imread(p)
        if img is None:
            rec.append({"file": p.name, "status": "unreadable"})
            continue
        with Timer() as t:
            faces = aligner.detector.detect(img)
        with Timer() as t2:
            afs = aligner.run(img)

        vis = img.copy()
        for a in afs:
            vis = draw_face_box(vis, a.face.bbox, a.face.landmarks,
                                label=f"{a.face.score:.2f}")
        h = 360
        vis = cv2.resize(vis, (int(vis.shape[1] * h / vis.shape[0]), h))
        det_tiles.append(vis)
        titles.append(p.stem.split("_")[-2] if "_" in p.stem else p.stem)

        item = {
            "file": p.name, "size": [img.shape[1], img.shape[0]],
            "n_faces": len(faces), "scores": [round(f.score, 4) for f in faces],
            "det_ms": round(t.ms, 2), "det_align_ms": round(t2.ms, 2),
            "swapped": [a.swapped for a in afs],
            "eye_dy": [round(a.eye_dy, 2) for a in afs],
            "face_wh": [[int(f.w), int(f.h)] for f in faces],
        }
        rec.append(item)

        if afs:
            for a in sorted(afs, key=lambda x: -x.face.area)[:1]:
                crop_tiles.append(a.crop)

    if crop_tiles:
        imwrite(out_dir / "alignment_grid.jpg",
                make_grid(crop_tiles, cols=cols, titles=titles[:len(crop_tiles)]))
    if det_tiles:
        imwrite(out_dir / "detection_grid.jpg", make_grid(det_tiles, cols=cols))

    ok = [r for r in rec if r.get("n_faces", 0) > 0]
    stats = {
        "n_images": len(rec),
        "n_with_face": len(ok),
        "detect_rate": round(len(ok) / max(1, len(rec)), 4),
        "avg_det_ms": round(float(np.mean([r["det_ms"] for r in ok])), 2) if ok else 0,
        "avg_det_align_ms": round(float(np.mean([r["det_align_ms"] for r in ok])), 2) if ok else 0,
        "n_swapped": int(sum(sum(r.get("swapped", [])) for r in rec)),
        "eye_dy_mean": round(float(np.mean([d for r in rec for d in r.get("eye_dy", [])])), 3)
        if ok else 0,
        "eye_dy_max": round(float(np.max([d for r in rec for d in r.get("eye_dy", [])])), 3)
        if ok else 0,
        "avg_score": round(float(np.mean([s for r in rec for s in r.get("scores", [])])), 4)
        if ok else 0,
        "per_image": rec,
    }
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", default=None, help="yunet | centerface")
    ap.add_argument("--no-lfw", action="store_true", help="跳过 LFW 批量对齐")
    ap.add_argument("--limit", type=int, default=0, help="每个身份最多处理多少张（0=全部）")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    out_dir = ensure_dir(CFG.output_dir / "step2")
    print("=" * 62)
    print(" Step 2  检测 + 对齐")
    print("=" * 62 + "\n")

    det = build_detector(args.detector, prefer_gpu=args.gpu)
    aligner = FaceAligner(det, crop_size=CFG.align.crop_size,
                          align_type=CFG.align.align_type,
                          auto_fix_reflection=CFG.align.auto_fix_reflection)
    print(f"检测器: {det}")
    print(f"对齐  : crop={aligner.crop_size}x{aligner.crop_size}, "
          f"transform={aligner.align_type}, 镜像自纠正={aligner.auto_fix_reflection}")
    print(f"参考点:\n{np.round(aligner.ref_pts, 3)}\n")

    # ---------------- 1. 本人照片
    print("[1] 本人照片 检测 + 对齐")
    stats = process_personal(aligner, out_dir)
    if stats:
        print(f"    图片数 {stats['n_images']}，检出 {stats['n_with_face']} "
              f"（检测率 {stats['detect_rate']*100:.1f}%）")
        print(f"    平均检测 {stats['avg_det_ms']:.1f} ms，检测+对齐 "
              f"{stats['avg_det_align_ms']:.1f} ms")
        print(f"    平均置信度 {stats['avg_score']:.3f}，"
              f"两眼水平差 均值 {stats['eye_dy_mean']:.2f}px / 最大 {stats['eye_dy_max']:.2f}px")
        print(f"    触发左右互换的图: {stats['n_swapped']} 张（应为 0，>0 说明检测器关键点语义不同）")
        (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
        print(f"    → {out_dir/'detection_grid.jpg'} / alignment_grid.jpg / stats.json")

    # ---------------- 2. LFW 批量对齐
    if not args.no_lfw:
        print("\n[2] LFW 公开库 批量对齐（供训练/评估）")
        for split in ("train", "valid"):
            src = CFG.lfw_dir / split
            if not src.exists():
                print(f"    [skip] {src} 不存在")
                continue
            t0 = time.perf_counter()
            st = align_folder(src, CFG.lfw_aligned_dir / split, aligner,
                              limit_per_id=args.limit,
                              save_fail_log=out_dir / f"align_fail_{split}.txt")
            dt = time.perf_counter() - t0
            print(f"    {split:<6} 共 {st['total']:>5} 张 → 成功 {st['ok']:>5}，"
                  f"失败 {st['fail']:>3}，多人脸 {st['multi']:>4}，互换 {st['swapped']:>3}，"
                  f"耗时 {dt:.1f}s")
            if st["fail"]:
                print(f"           失败清单: {out_dir/f'align_fail_{split}.txt'}")

    print("\n✓ Step 2 完成。下一步： python scripts/step3_build_gallery.py")


if __name__ == "__main__":
    main()

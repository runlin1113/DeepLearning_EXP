"""
Step 1 —— 环境自检
==================
检查：Python / 依赖版本 / GPU / 模型文件 / 端到端冒烟测试（检测→对齐→特征→比对）。

用法：
    python scripts/step1_env_check.py
    python scripts/step1_env_check.py --gpu      # 尝试用 CUDA 跑 onnxruntime
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true", help="优先使用 CUDAExecutionProvider")
    ap.add_argument("--image", default=None, help="冒烟测试用图，默认取本人第一张照片")
    args = ap.parse_args()

    print("=" * 62)
    print(" Step 1  环境自检")
    print("=" * 62)

    # ---------- 1. Python / 系统
    print(f"[1] Python {platform.python_version()}  |  {platform.system()} {platform.machine()}")
    if sys.version_info < (3, 9):
        print("    ✗ 建议 Python >= 3.9")

    # ---------- 2. 依赖
    print("\n[2] 关键依赖")
    req = {
        "numpy": "1.24", "cv2": "4.8", "onnxruntime": "1.15",
        "torch": None, "PIL": "9.0", "matplotlib": "3.5", "tqdm": "4.60",
    }
    missing = []
    for mod, _min in req.items():
        name = "Pillow" if mod == "PIL" else ("opencv-python" if mod == "cv2" else mod)
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"    ✓ {name:<12} {ver}")
        except Exception:
            print(f"    ✗ {name:<12} 未安装")
            missing.append(name)
    if missing:
        print(f"    → pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "
              f"{' '.join(missing)}")

    # ---------- 3. GPU
    print("\n[3] 硬件加速")
    try:
        import torch
        print(f"    torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    GPU: {torch.cuda.get_device_name(0)}")
    except Exception as e:
        print(f"    torch 不可用: {e}")
    import onnxruntime as ort
    print(f"    onnxruntime {ort.__version__}  providers={ort.get_available_providers()}")

    # ---------- 4. 模型
    print("\n[4] 模型文件")
    from face_system.config import CFG
    for label, p, need in [("YuNet 检测器", CFG.detector.yunet_model, 100_000),
                           ("ArcFace 骨干", CFG.rec.model, 100_000_000)]:
        p = Path(p)
        ok = p.exists() and p.stat().st_size >= need
        print(f"    {'✓' if ok else '✗'} {label:<14} {p.name} "
              f"({p.stat().st_size/1e6:.2f} MB)" if p.exists()
              else f"    ✗ {label:<14} 缺失 {p}")
        if not ok:
            print("      → 先运行 python scripts/step0_download_models.py")

    # ---------- 5. 数据
    print("\n[5] 数据")
    from face_system.utils import list_images
    for label, d in [("本人照片", CFG.raw_face_dir), ("LFW 公开库", CFG.lfw_dir)]:
        n = len(list_images(d, recursive=True)) if Path(d).exists() else 0
        print(f"    {'✓' if n else '✗'} {label:<10} {n:>5} 张  {d}")

    # ---------- 6. 端到端冒烟
    print("\n[6] 端到端冒烟测试")
    imgs = list_images(CFG.raw_face_dir)
    img_path = args.image or (str(imgs[0]) if imgs else None)
    if not img_path:
        print("    ✗ 没有可用的测试图")
        return

    import numpy as np
    from face_system import build_pipeline
    from face_system.utils import imread, cosine_sim

    t0 = time.perf_counter()
    det, aligner, rec = build_pipeline(prefer_gpu=args.gpu)
    print(f"    detector : {det}")
    print(f"    recognizer: {rec}")
    print(f"    加载耗时 {1000*(time.perf_counter()-t0):.0f} ms")

    img = imread(img_path)
    print(f"    测试图: {Path(img_path).name}  {img.shape[1]}x{img.shape[0]}")

    t0 = time.perf_counter()
    afs = aligner.run(img)
    t_det = 1000 * (time.perf_counter() - t0)
    print(f"    检测+对齐: {len(afs)} 张人脸, {t_det:.1f} ms")
    if not afs:
        print("    ✗ 未检出人脸")
        return
    best = max(afs, key=lambda a: a.face.area)
    print(f"    对齐输出: {best.crop.shape}  双眼水平差={best.eye_dy:.2f}px  "
          f"左右互换={best.swapped}")

    f = rec.get_feat([a.crop for a in afs])
    print(f"    特征: {f.shape}  L2范数={np.linalg.norm(f, axis=1).round(4)}  "
          f"{rec.avg_ms:.1f} ms/张")
    print(f"    自比对余弦相似度: {cosine_sim(f[0], f[0])[0,0]:.4f} (应=1.0000)")

    out = ROOT / "outputs" / "step1_smoke"
    out.mkdir(parents=True, exist_ok=True)
    from face_system.utils import imwrite, draw_face_box
    vis = img.copy()
    for a in afs:
        vis = draw_face_box(vis, a.face.bbox, a.face.landmarks,
                            label=f"face {a.face.score:.2f}")
    imwrite(out / "detect.jpg", vis)
    imwrite(out / "aligned.jpg", best.crop)
    print(f"\n    可视化已保存: {out}")
    print("\n✓ 环境自检完成，可以进入 Step 2。")


if __name__ == "__main__":
    main()

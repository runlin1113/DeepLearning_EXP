"""
Step 0 —— 模型下载与校验
=========================
本实验用到的两个模型（共约 175MB）：
    1) YuNet 检测器   face_detection_yunet_2023mar.onnx   (~230KB)
    2) ArcFace 骨干   w600k_r50.onnx                      (~174MB, buffalo_l 压缩包内)

用法：
    python scripts/step0_download_models.py              # 缺什么下什么
    python scripts/step0_download_models.py --force      # 全部重下

国内网络建议先设置镜像（Windows git-bash）：
    export https_proxy=http://127.0.0.1:7890
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from face_system.config import CFG, MODEL_DIR  # noqa: E402

YUNET_URLS = [
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
]
BUFFALO_URLS = [
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
]
CENTERFACE_URLS = [
    "https://media.githubusercontent.com/media/Star-Clouds/CenterFace/master/models/centerface.onnx",
    "https://github.com/Star-Clouds/CenterFace/raw/master/models/centerface.onnx",
]


def sha256(path: Path, n: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def download(urls, dst: Path, min_size: int = 1024) -> bool:
    import urllib.request
    dst.parent.mkdir(parents=True, exist_ok=True)
    for u in urls:
        try:
            print(f"  ↓ {u}")
            tmp = dst.with_suffix(dst.suffix + ".part")
            with urllib.request.urlopen(u, timeout=60) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        print(f"\r    {got/1e6:.1f}/{total/1e6:.1f} MB", end="", flush=True)
            print()
            if tmp.stat().st_size < min_size:
                print(f"    ✗ 文件过小（{tmp.stat().st_size} B），可能是 LFS 指针页，换下一个源")
                tmp.unlink(missing_ok=True)
                continue
            tmp.replace(dst)
            print(f"    ✓ {dst.name} ({dst.stat().st_size/1e6:.2f} MB)")
            return True
        except Exception as e:
            print(f"    ✗ {type(e).__name__}: {e}")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="已存在也重新下载")
    ap.add_argument("--centerface", action="store_true", help="额外下载 CenterFace（可选）")
    args = ap.parse_args()

    print(f"模型目录: {MODEL_DIR}\n")

    # ---- 1) YuNet
    p = Path(CFG.detector.yunet_model)
    if p.exists() and not args.force:
        print(f"[1/2] YuNet 已存在: {p.name} ({p.stat().st_size} B)")
    else:
        print("[1/2] 下载 YuNet ...")
        if not download(YUNET_URLS, p, min_size=100_000):
            sys.exit("YuNet 下载失败，请手动放置到 " + str(p))
    print(f"      sha256[:16] = {sha256(p)}\n")

    # ---- 2) ArcFace (buffalo_l)
    rec = Path(CFG.rec.model)
    if rec.exists() and not args.force:
        print(f"[2/2] ArcFace 已存在: {rec.name} ({rec.stat().st_size/1e6:.1f} MB)")
    else:
        zpath = MODEL_DIR / "buffalo_l.zip"
        print("[2/2] 下载 buffalo_l（含 ArcFace w600k_r50）...")
        if not download(BUFFALO_URLS, zpath, min_size=100_000_000):
            sys.exit("buffalo_l 下载失败。可手动下载后解压，只保留 w600k_r50.onnx 放进 models/")
        print("      解压中 ...")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(MODEL_DIR)
        for junk in ("1k3d68.onnx", "2d106det.onnx", "det_10g.onnx", "genderage.onnx"):
            (MODEL_DIR / junk).unlink(missing_ok=True)
        zpath.unlink(missing_ok=True)
        print(f"      ✓ {rec.name} ({rec.stat().st_size/1e6:.1f} MB)")
    print(f"      sha256[:16] = {sha256(rec)}\n")

    # ---- 3) CenterFace（可选后端）
    if args.centerface:
        cp = MODEL_DIR / "centerface.onnx"
        if not cp.exists() or args.force:
            print("[可选] 下载 CenterFace ...")
            if download(CENTERFACE_URLS, cp, min_size=5_000_000):
                print("      注意：若本工程用 YuNet 则可忽略此模型")
            else:
                print("      CenterFace 下载失败（其仓库用 LFS，镜像常失效），"
                      "本实验默认 YuNet，不影响任何步骤")

    print("\n全部就绪。下一步： python scripts/step1_env_check.py")


if __name__ == "__main__":
    main()

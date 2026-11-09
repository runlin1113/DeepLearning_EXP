"""
Step 5 —— 视频动态人脸识别（追踪 + 多帧特征融合）
=================================================
输入：摄像头(0) / 视频文件 / 图片序列目录 / 合成测试视频
链路：帧 → 检测 → 对齐 → ArcFace 特征 → 轨迹关联 → EMA 多帧融合 → 底库检索 → 时序投票 → 画框输出

三种模式的对比（--mode）：
    frame          逐帧独立识别（不做轨迹，最朴素，会闪烁）
    track_nofusion 有轨迹，但只用当前帧特征识别（把 ema_alpha 设为 1）
    track_fusion   轨迹 + EMA 融合 + 投票（本实验方案，默认）

用法：
    python scripts/step5_video_demo.py --make-test-video          # 先生成测试视频
    python scripts/step5_video_demo.py                            # 用默认测试视频跑
    python scripts/step5_video_demo.py --source 0                 # 摄像头
    python scripts/step5_video_demo.py --source my.mp4 --mode frame
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from face_system import (ArcFaceONNX, FaceAligner, Gallery, TrackManager,  # noqa: E402
                         build_detector)
from face_system.config import CFG, ensure_dir  # noqa: E402
from face_system.utils import (FPSMeter, draw_face_box, imread, imwrite,  # noqa: E402
                               list_images, make_grid, put_text_scale)

OUT = ensure_dir(CFG.output_dir / "step5")
GAL_DIR = CFG.output_dir / "gallery"
TEST_VIDEO = OUT / "test_video.mp4"

COLORS = [(80, 220, 80), (80, 160, 255), (255, 120, 120), (220, 120, 255),
          (120, 255, 220), (255, 200, 80), (200, 200, 120)]


# ============================================================ 合成测试视频
def _ellipse_paste(canvas, face_bgr, center, size):
    """把对齐人脸以椭圆羽化蒙版贴到画布上，模拟"人在画面里"。"""
    w, h = size
    face = cv2.resize(face_bgr, (w, h), interpolation=cv2.INTER_AREA)
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.46), int(h * 0.46)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1, w // 24))
    mask = mask[..., None]
    cx, cy = int(center[0]), int(center[1])
    x1, y1 = cx - w // 2, cy - h // 2
    H, W = canvas.shape[:2]
    x2, y2 = min(W, x1 + w), min(H, y1 + h)
    x1c, y1c = max(0, x1), max(0, y1)
    if x2 <= x1c or y2 <= y1c:
        return canvas
    fx1, fy1 = x1c - x1, y1c - y1
    sub = mask[fy1:fy1 + (y2 - y1c), fx1:fx1 + (x2 - x1c)]
    fsub = face[fy1:fy1 + (y2 - y1c), fx1:fx1 + (x2 - x1c)]
    region = canvas[y1c:y2, x1c:x2].astype(np.float32)
    canvas[y1c:y2, x1c:x2] = (region * (1 - sub) + fsub.astype(np.float32) * sub).astype(np.uint8)
    return canvas


def _degrade(face, rng, k: int = 0, hard: bool = False):
    """给贴图人脸加"真实摄像机"退化：降采样、运动模糊、光照漂移、局部遮挡。
    没有退化的话，逐帧识别几乎是 100% 稳定，看不出追踪+融合的价值。"""
    h, w = face.shape[:2]
    # 1) 分辨率下降（远处小脸）
    scale = (0.62 + 0.08 * np.sin(k * 0.7)) if hard else (0.72 + 0.10 * np.sin(k * 0.7))
    face = cv2.resize(face, (max(24, int(w * scale)), max(24, int(h * scale))),
                      interpolation=cv2.INTER_AREA)
    # 2) 运动模糊（偶数帧）
    if k % 3 == 0:
        ker = np.zeros((5, 5), np.float32)
        ker[2, :] = 1.0 / 5
        face = cv2.filter2D(face, -1, ker)
    face = cv2.GaussianBlur(face, (0, 0), 1.0 if hard else 0.6)
    # 1.5) 头部小幅度转动/俯仰（用仿射近似），模拟真实视频里的姿态变化
    if hard:
        ang = 4.0 * np.sin(k * 0.9) + rng.normal(0, 1.5)
        sc = 1.0 + rng.normal(0, 0.03)
        M = cv2.getRotationMatrix2D((face.shape[1] / 2, face.shape[0] / 2), ang, sc)
        face = cv2.warpAffine(face, M, (face.shape[1], face.shape[0]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # 1.6) 视频压缩伪影
    if hard:
        ok, enc = cv2.imencode(".jpg", face, [int(cv2.IMWRITE_JPEG_QUALITY), 35])
        if ok:
            face = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    # 3) 亮度/噪声漂移
    face = np.clip(face.astype(np.float32) * (0.88 + 0.18 * rng.random()) +
                   rng.normal(0, 4, face.shape), 0, 255).astype(np.uint8)
    # 4) 偶尔遮挡（模拟手/麦克风挡住下半脸）
    if k % 7 == 3:
        oh, ow = int(face.shape[0] * 0.34), int(face.shape[1] * 0.6)
        y0 = face.shape[0] - oh
        x0 = (face.shape[1] - ow) // 2
        face[y0:, x0:x0 + ow] = (60, 60, 60)
    return face


def make_test_video(path: Path = TEST_VIDEO, seconds: int = 10, fps: int = 25,
                    W: int = 960, H: int = 540, degrade: bool = True,
                    hard: bool = False):
    """
    合成一段"多人依次入场"的测试视频：
        A = 本人（已注册）      左右往返移动
        B = Arnold_Schwarzenegger（已注册）   纵向上下移动，第 2 秒入场
        C = Kofi_Annan（**未注册**，应被判为"未知"）  第 5 秒从右侧进入
    这样一段视频同时验证：多人追踪、已注册识别、陌生人拒识、特征融合抗抖动。
    """
    print("[make-test-video] 合成测试视频 ...")
    aligned_train = CFG.lfw_aligned_dir / "train"
    split = json.loads((GAL_DIR / "self_split.json").read_text(encoding="utf-8"))
    me = [imread(CFG.raw_face_dir / n) for n in split["probe"][:4]]
    me = [c for c in me if c is not None]
    # 本人探测图重新对齐（原始图直接缩放会变形，这里用对齐图当素材）
    det = build_detector()
    aligner = FaceAligner(det)
    me_crops = []
    for p in [CFG.raw_face_dir / n for n in split["probe"][:4]]:
        img = imread(p)
        if img is None:
            continue
        afs = aligner.run(img)
        if afs:
            me_crops.append(max(afs, key=lambda a: a.face.area).crop)
    if not me_crops:
        raise SystemExit("合成视频失败：没有可用的本人对齐人脸")

    def lfw_crops(name, k=3):
        imgs = list_images(aligned_train / name)[:k]
        return [imread(p) for p in imgs if imread(p) is not None]

    other = lfw_crops("Arnold_Schwarzenegger", 3)
    unknown = lfw_crops("Kofi_Annan", 3)
    if not other or not unknown:
        raise SystemExit("合成视频失败：LFW 素材缺失（先跑 step2）")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    n = seconds * fps
    rng = np.random.default_rng(0)

    # 背景：柔和渐变 + 轻微纹理，避免纯色导致检测器行为过于理想
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = np.stack([80 + 60 * yy / H, 95 + 45 * xx / W, 120 + 30 * yy / H], -1)

    for i in range(n):
        t = i / fps
        frame = bg.copy()
        frame += rng.normal(0, 3, frame.shape)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # A：本人，水平往返
        cx = W * 0.32 + W * 0.20 * np.sin(2 * np.pi * t / 5.0)
        sz = int((100 if hard else 118) * (1 + 0.10 * np.sin(2 * np.pi * t / 3.0)))
        fa = me_crops[int(t * 2) % len(me_crops)]
        if degrade:
            fa = _degrade(fa, rng, i, hard)
        frame = _ellipse_paste(frame, fa, (cx, H * 0.52), (sz, sz))

        # B：已注册的 LFW 身份，第 2 秒入场，纵向移动
        if t >= 2.0:
            cy = H * 0.50 + H * 0.16 * np.sin(2 * np.pi * (t - 2) / 4.0)
            sz2 = 92 if hard else 104
            fb = other[int(t * 1.5) % len(other)]
            if degrade:
                fb = _degrade(fb, rng, i + 1, hard)
            frame = _ellipse_paste(frame, fb, (W * 0.68, cy), (sz2, sz2))

        # C：未注册的陌生人，第 5 秒入场，从右侧滑入
        if t >= 5.0:
            k = min(1.0, (t - 5.0) / 1.5)
            cx3 = W * (1.15 - 0.30 * k)
            sz3 = 88 if hard else 100
            fc = unknown[int(t * 1.5) % len(unknown)]
            if degrade:
                fc = _degrade(fc, rng, i + 2, hard)
            frame = _ellipse_paste(frame, fc, (cx3, H * 0.42), (sz3, sz3))

        put_text_scale(frame, f"t={t:04.1f}s  SYNTHETIC TEST", (10, 8),
                       color=(255, 255, 255), size=18)
        vw.write(frame)
    vw.release()
    print(f"    → {path}  ({n} 帧, {W}x{H}, {fps}fps, {seconds}s)")
    return path


# ================================================================= 识别主循环
def run_video(source, gallery: Gallery, thr: float, mode: str = "track_fusion",
              max_frames: int = 0, save_video: Path | None = None,
              gpu: bool = False, feat_interval: int = 3):
    det = build_detector(prefer_gpu=gpu)
    aligner = FaceAligner(det, crop_size=CFG.align.crop_size)
    rec = ArcFaceONNX(CFG.rec.model, prefer_gpu=gpu)

    # track_fusion  : EMA 融合 + 时序投票（本实验方案）
    # track_nofusion: 只保留轨迹，特征与标签都用当前帧（对照）
    # frame         : 连轨迹都不要，纯逐帧识别（基线）
    if mode == "track_fusion":
        alpha, min_hits, vote_window = CFG.rec.ema_alpha, CFG.rec.min_track_hits, CFG.rec.vote_window
    elif mode == "track_nofusion":
        alpha, min_hits, vote_window = 1.0, 1, 1
    else:
        alpha, min_hits, vote_window = 1.0, 1, 1
    manager = TrackManager(iou_thr=0.3, max_lost=CFG.rec.track_buffer,
                           ema_alpha=alpha, min_hits=min_hits,
                           vote_window=vote_window)

    cap = cv2.VideoCapture(source if isinstance(source, (int, str)) else str(source))
    if not cap.isOpened():
        raise SystemExit(f"打不开视频源: {source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vw = None
    if save_video:
        vw = cv2.VideoWriter(str(save_video), cv2.VideoWriter_fourcc(*"mp4v"),
                             max(1.0, fps_in), (W, H))
    print(f"    源: {source}  {W}x{H} @{fps_in:.0f}fps  mode={mode}  τ={thr:.2f}")

    meter = FPSMeter(30)
    t_start = time.perf_counter()
    frame_idx = 0
    per_frame = []
    track_hist = defaultdict(list)          # tid -> [(frame, name, sim)]
    prev_dets = []                          # frame 模式下的相邻帧匹配用
    flick = {"matched": 0, "switch": 0}

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if max_frames and frame_idx > max_frames:
            break

        t0 = time.perf_counter()
        faces = det.detect(frame)
        # 关键帧策略：每隔 feat_interval 帧才做 ArcFace（最贵的一步），
        # 中间帧复用轨迹里 EMA 累积的特征 —— 这是实时系统的常规做法。
        need_feat = (mode == "frame") or (feat_interval <= 1) or (frame_idx % feat_interval == 0)
        if need_feat and faces:
            crops, keep = [], []
            for j, f in enumerate(faces):
                try:
                    af = aligner.align_face(frame, f)
                except Exception:
                    continue
                crops.append(af.crop)
                keep.append(j)
            feats = rec.get_feat(crops) if crops else np.zeros((0, 512), np.float32)
            faces = [faces[j] for j in keep]
        else:
            feats = [None] * len(faces)

        if mode == "frame":
            # 无轨迹：逐帧独立识别（基线）。用"最近邻位置匹配"统计相邻帧标签跳变率，
            # 因为逐帧系统本身没有 track id，直接比较同名/异名更公平。
            labels = []
            for i in range(len(faces)):
                name, sim, _ = gallery.identify(feats[i], threshold=thr)
                labels.append((name, sim))
            cur = []
            for i, f in enumerate(faces):
                cur.append(((f.x + f.w / 2, f.y + f.h / 2), 0.5 * (f.w + f.h), labels[i][0]))
            used = set()
            for (c, s, nm) in cur:
                best, bd = -1, 1e9
                for j, (pc, ps, pn) in enumerate(prev_dets):
                    if j in used:
                        continue
                    d = float(np.hypot(c[0] - pc[0], c[1] - pc[1]))
                    if d < 0.6 * ps and d < bd:
                        bd, best = d, j
                if best >= 0:
                    used.add(best)
                    flick["matched"] += 1
                    flick["switch"] += int(prev_dets[best][2] != nm)
            prev_dets = cur
        else:
            tracks = manager.step(faces, feats, gallery, threshold=thr)
            labels = []
            for t in tracks:
                labels.append((t.name, t.sim))
                track_hist[t.tid].append((frame_idx, t.name, t.sim))
                per_frame.append({"frame": frame_idx, "tid": t.tid, "name": t.name,
                                  "sim": round(float(t.sim), 4),
                                  "bbox": [round(float(v), 1) for v in t.bbox]})

        # ---- 绘制
        if mode == "frame":
            for i, f in enumerate(faces):
                name, sim = labels[i]
                col = (60, 220, 60) if name != "未知" else (60, 60, 230)
                frame = draw_face_box(frame, f.bbox, f.landmarks, color=col,
                                      label=f"{name} {sim:.2f}", label_size=22)
        else:
            for t in tracks:
                col = COLORS[t.tid % len(COLORS)]
                if t.name in ("未知",):
                    col = (60, 60, 230)
                elif t.name in ("识别中", ""):
                    col = (200, 200, 200)
                frame = draw_face_box(frame, t.bbox, None, color=col,
                                      label=f"#{t.tid} {t.name} {t.sim:.2f}", label_size=22)
                pts = np.array(t.trail, np.int32).reshape(-1, 1, 2)
                if len(pts) > 2:
                    cv2.polylines(frame, [pts], False, col, 1, cv2.LINE_AA)

        fps = meter.tick()
        put_text_scale(frame, f"FPS {fps:5.1f} | {mode} | thr {thr:.2f} | frame {frame_idx}",
                       (10, 8), color=(255, 255, 255), size=16)
        if vw is not None:
            vw.write(frame)

    cap.release()
    if vw is not None:
        vw.release()
    total_time = time.perf_counter() - t_start

    # ---- 轨迹汇总
    track_summary = []
    for tid, hist in track_hist.items():
        names = [n for _, n, _ in hist if n not in ("识别中", "")]
        if not names:
            continue
        final = Counter(names).most_common(1)[0][0]
        sims = [s for _, n, s in hist if n == final]
        switches = sum(1 for a, b in zip(names, names[1:]) if a != b)
        track_summary.append({
            "tid": int(tid), "frames": len(hist), "final_label": final,
            "mean_sim": round(float(np.mean(sims)), 4) if sims else 0.0,
            "stability": round(float(np.mean([n == final for n in names])), 4),
            "label_switches": int(switches),
        })
    track_summary.sort(key=lambda r: -r["frames"])

    stats = {
        "source": str(source), "mode": mode, "threshold": thr,
        "frames": frame_idx, "fps_proc": round(frame_idx / max(1e-6, total_time), 2),
        "avg_frame_ms": round(1000 * total_time / max(1, frame_idx), 2),
        "n_tracks": len(track_summary),
        "tracks": track_summary,
        "mean_stability": round(float(np.mean([t["stability"] for t in track_summary])), 4)
        if track_summary else round(1 - flick["switch"] / max(1, flick["matched"]), 4),
        "total_switches": int(sum(t["label_switches"] for t in track_summary))
        if track_summary else flick["switch"],
        "flicker_matched_pairs": flick["matched"],
    }
    return stats, per_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="视频路径 / 0=摄像头 / 省略=合成视频")
    ap.add_argument("--make-test-video", action="store_true")
    ap.add_argument("--clean-video", action="store_true", help="合成视频不加退化（更容易的版本）")
    ap.add_argument("--hard-video", action="store_true",
                    help="合成视频加远距离+压缩+姿态抖动退化，用于凸显追踪融合的作用")
    ap.add_argument("--gallery", default=str(GAL_DIR / "gallery_demo.npz"))
    ap.add_argument("--mode", default="track_fusion",
                    choices=["track_fusion", "track_nofusion", "frame"])
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--feat-interval", type=int, default=3,
                    help="每隔多少帧提取一次 ArcFace 特征（1=每帧，最慢）")
    ap.add_argument("--save-frames", type=int, default=0, help="额外导出 N 张关键帧拼图")
    args = ap.parse_args()

    print("=" * 62)
    print(" Step 5  视频动态识别")
    print("=" * 62 + "\n")

    if args.make_test_video or (args.source is None and not TEST_VIDEO.exists()):
        make_test_video(degrade=not args.clean_video, hard=args.hard_video)

    src = 0 if args.source == "0" else (args.source or str(TEST_VIDEO))
    gallery = Gallery.load(args.gallery).finalize()
    thr = args.thr if args.thr is not None else gallery.threshold

    out_video = OUT / f"annotated_{args.mode}.mp4"
    stats, per_frame = run_video(src, gallery, thr, mode=args.mode,
                                 max_frames=args.max_frames, save_video=out_video,
                                 gpu=args.gpu, feat_interval=args.feat_interval)
    print(f"\n    处理 {stats['frames']} 帧，平均 {stats['avg_frame_ms']:.1f} ms/帧 "
          f"({stats['fps_proc']:.1f} FPS)")
    print(f"    轨迹数 {stats['n_tracks']}，平均标签稳定度 {stats['mean_stability']*100:.1f}%，"
          f"标签切换次数 {stats['total_switches']}")
    for t in stats["tracks"]:
        print(f"      #{t['tid']:<2} {t['final_label']:<20} 帧数 {t['frames']:<4} "
              f"平均相似度 {t['mean_sim']:.3f}  稳定度 {t['stability']*100:.1f}%")

    (OUT / f"video_stats_{args.mode}.json").write_text(
        json.dumps({"stats": stats, "per_frame": per_frame[:2000]},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n    → 标注视频 {out_video}")
    print(f"    → 统计 {OUT/f'video_stats_{args.mode}.json'}")

    # 关键帧拼图
    if out_video.exists():
        cap = cv2.VideoCapture(str(out_video))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        picks = np.linspace(0, max(0, total - 1), 8).astype(int) if total > 0 else []
        tiles = []
        for p in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(p))
            ok, fr = cap.read()
            if ok:
                tiles.append(cv2.resize(fr, (640, 360)))
        cap.release()
        if tiles:
            imwrite(OUT / f"video_frames_{args.mode}.jpg", make_grid(tiles, cols=2))
            print(f"    → 关键帧拼图 {OUT/f'video_frames_{args.mode}.jpg'}")
    print("\n✓ Step 5 完成。建议再跑 --mode frame / track_nofusion 做对比。")


if __name__ == "__main__":
    main()

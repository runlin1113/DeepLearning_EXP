"""
刷脸打卡（Face Attendance）—— 实验七流水线的落地应用
=====================================================

应用场景
--------
校田径队早上训练，队员走到门口的摄像头前，系统自动认出是谁、几点到的，
写进当天的考勤表。替代"排队按指纹/刷卡"：不用接触、不用停留、认不出的人
（陌生人、家属、路过的同学）直接忽略，不产生任何记录。

它其实把实验七的五段链路**原封不动**用了一遍：

    摄像头帧 → ① 检测（YuNet）→ ② 对齐（112×112）
             → ③ 特征（ArcFace 512 维）→ ④ 比对（与底库做 1:N 余弦检索）
             → ⑤ 时序（追踪 + EMA 融合 + 投票）→ 打卡记录 CSV

和"逐帧识别"相比，多出来的 ⑤ 正是本实验的重点：人在门口走动的十几帧里，
检测框会抖、偶尔侧脸、偶尔被后面的人挡住。逐帧独立识别会出现"张三→未知→张三"
这种闪烁，落到考勤上就是重复打卡或漏打卡。

用法
----
    # 跑测试视频（工程里已有的合成视频，250 帧，含本人 + 陌生人）
    python app/attendance_demo.py --video outputs/step5/test_video.mp4

    # 接本机摄像头（按 q 退出）
    python app/attendance_demo.py --camera 0

    # 只跑前 100 帧、每 5 帧提一次特征（更快）
    python app/attendance_demo.py --video xxx.mp4 --max-frames 100 --feat-interval 5

产出
----
    outputs/app/attendance_YYYYmmdd_HHMMSS.csv   打卡记录
    outputs/app/attendance_demo.mp4              带标注的视频（便于课堂演示）

深度学习课程实验七 · 人脸识别应用示例
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

# 让脚本能 import 到工程根目录下的 face_system 包
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from face_system import build_pipeline                      # noqa: E402
from face_system.gallery import Gallery                     # noqa: E402
from face_system.tracker import TrackManager                # noqa: E402
from face_system.utils import draw_face_box, put_text_scale  # noqa: E402

# --------------------------------------------------------------------------
# 配置：这些值都对应实验里调过的超参，注释标了"为什么是这个值"
# --------------------------------------------------------------------------
GALLERY_PATH = ROOT / "outputs" / "gallery" / "gallery_demo.npz"
OUT_DIR = ROOT / "outputs" / "app"

THRESHOLD = 0.35        # 余弦阈值：≥0.35 认作库里的人，否则"未知"
                        # 部署值比 EER 最优值(0.236)高，因为考勤"认错人"比"认不出"更糟
EMA_ALPHA = 0.6         # 特征融合系数：新特征占 0.6，历史特征占 0.4
VOTE_WINDOW = 15        # 投票窗口：最近 15 次识别结果做多数票
MIN_HITS = 3            # 至少看到 3 次才给身份，避免刚进画面就误判
MAX_LOST = 30           # 轨迹连续丢失 30 帧才删除（约 1 秒，容忍短暂遮挡）
FEAT_INTERVAL = 3       # 每 3 帧提一次特征，中间帧复用 EMA 结果（ArcFace 是瓶颈）


def load_gallery() -> Gallery:
    """加载注册底库。底库里存的是每个身份的若干张 512 维特征（已 L2 归一化）。"""
    if not GALLERY_PATH.exists():
        raise FileNotFoundError(
            f"底库不存在：{GALLERY_PATH}\n请先跑 python scripts/step3_build_gallery.py")
    g = Gallery.load(GALLERY_PATH)
    print(g.summary())
    return g


def process_frame(frame, det, aligner, rec, tracker, gallery, do_feat: bool):
    """处理一帧：检测 → 对齐 → (特征) → 追踪融合投票。

    do_feat=False 时只做几何部分（检测 + 对齐），特征传 None，
    轨迹会沿用上一帧的 EMA 特征——这是 16 FPS 的关键。
    """
    # ---- ① 检测：得到 N 个 Face(bbox, score, landmarks)
    #      注意 det.detect 内部已经做过 letterbox 缩放与 NMS
    faces = det.detect(frame)

    feats = None
    if faces and do_feat:
        # ---- ②③ 对齐 + 特征：只在关键帧做，一次批量送进网络
        crops = []
        for f in faces:
            try:
                crops.append(aligner.align_face(frame, f).crop)   # 112×112×3
            except Exception:
                crops.append(None)                                # 单张失败不拖累整帧
        valid = [(i, c) for i, c in enumerate(crops) if c is not None]
        if valid:
            idxs, imgs = zip(*valid)
            batch = rec.get_feat(list(imgs))                      # (M, 512)，已归一化
            feats = [None] * len(faces)
            for k, i in enumerate(idxs):
                feats[i] = batch[k]
            feats = [f if f is not None else None for f in feats]

    # ---- ⑤ 追踪 + 融合 + 投票：内部会做 IoU 关联、EMA 融合和多数票
    tracks = tracker.step(faces, feats, gallery, threshold=THRESHOLD)
    return faces, tracks


def main():
    ap = argparse.ArgumentParser(description="刷脸打卡 demo")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", help="视频文件路径")
    src.add_argument("--camera", type=int, help="摄像头编号，一般为 0")
    ap.add_argument("--max-frames", type=int, default=0, help="最多处理多少帧（0=全部）")
    ap.add_argument("--feat-interval", type=int, default=FEAT_INTERVAL)
    ap.add_argument("--no-video", action="store_true", help="不写标注视频（只看 CSV）")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 准备三个模型 ----------
    print("加载模型（YuNet 检测 + evoLVe 对齐 + ArcFace 特征）…")
    det, aligner, rec = build_pipeline()
    gallery = load_gallery()
    tracker = TrackManager(ema_alpha=EMA_ALPHA, min_hits=MIN_HITS,
                           vote_window=VOTE_WINDOW, max_lost=MAX_LOST)

    # ---------- 打开输入 ----------
    source = args.camera if args.camera is not None else args.video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"打不开输入：{source}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ---------- 准备输出 ----------
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"attendance_{stamp}.csv"
    vid_path = OUT_DIR / "attendance_demo.mp4"
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_in, (W, H))

    # ---------- 打卡状态 ----------
    punched: dict[str, dict] = {}     # 姓名 → 首次打卡信息（同一次运行只记一次）
    rows = []                          # 最终写进 CSV 的行
    fi = 0
    t0 = time.time()

    print(f"开始处理：{source}  {W}×{H} @ {fps_in:.1f} fps")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 关键帧策略：每 feat_interval 帧才跑一次 ArcFace
        do_feat = (fi % max(1, args.feat_interval) == 0)
        faces, tracks = process_frame(frame, det, aligner, rec, tracker,
                                      gallery, do_feat)

        # ---------- 画框 + 触发打卡 ----------
        for t in tracks:
            name, sim = t.name, t.sim
            known = name not in ("识别中", "未知", "")
            color = (0, 200, 0) if known else (0, 165, 255)   # 绿=已注册，橙=陌生人
            label = f"{name} {sim:.2f}" if known else name
            draw_face_box(frame, t.bbox, color=color, label=label, label_size=20)

            # 只有"已确认身份 + 本帧是新特征 + 本次运行还没记过"才打卡
            if known and t.feat_fresh and name not in punched:
                punched[name] = {
                    "姓名": name,
                    "打卡时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "相似度": round(float(sim), 4),
                    "轨迹ID": t.tid,
                    "触发帧号": fi,
                }
                rows.append(punched[name])
                put_text_scale(frame, f"✓ {name} 打卡成功", (12, 12),
                               color=(0, 200, 0), size=26)
                print(f"  [打卡] 帧 {fi:>4}  {name:<20s} sim={sim:.3f}  tid={t.tid}")

        # ---------- 角落信息 ----------
        info = f"frame {fi}  tracks {len(tracks)}  punched {len(punched)}"
        put_text_scale(frame, info, (12, H - 34), color=(240, 240, 240), size=18)

        if writer is not None:
            writer.write(frame)
        fi += 1
        if args.max_frames and fi >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    # ---------- 落盘 ----------
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["姓名", "打卡时间", "相似度", "轨迹ID", "触发帧号"])
        w.writeheader()
        w.writerows(rows)

    dt = time.time() - t0
    print("\n" + "=" * 56)
    print(f"处理 {fi} 帧，耗时 {dt:.1f}s（{fi / max(dt, 1e-6):.1f} FPS）")
    print(f"打卡 {len(rows)} 人：{', '.join(r['姓名'] for r in rows) or '（无）'}")
    print(f"考勤表：{csv_path}")
    if writer is not None:
        print(f"标注视频：{vid_path}")


if __name__ == "__main__":
    main()

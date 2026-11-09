"""
视频动态识别：人脸追踪（IoU + 匀速预测匹配）+ 多帧特征融合 + 时序投票。

为什么需要它？
    逐帧独立识别会抖动（同一人一会儿 A 一会儿 B），且单帧侧脸/模糊时特征不可靠。
    做法：
    1) 用 IoU 把本帧检测框和历史轨迹关联起来，得到稳定的 track id；
    2) 每帧新特征与轨迹的历史特征做 EMA 融合（越靠后的帧权重越大）：
           f_t = normalize( α·f_new + (1-α)·f_{t-1} )
       等价于一个"指数加权的多帧特征平均"，方差按 1/√N_eff 下降；
    3) 融合特征去底库检索，再把最近 W 帧的 top-1 结果做**多数投票**，
       票数不足 min_track_hits 时显示"识别中"，避免闪烁。
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np

from .gallery import Gallery
from .utils import bbox_iou, center_dist, l2norm


@dataclass
class Track:
    tid: int
    bbox: np.ndarray                       # xywh
    score: float = 1.0
    feat: Optional[np.ndarray] = None      # EMA 融合特征（已归一化）
    hits: int = 1
    age: int = 1
    lost: int = 0
    vel: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    votes: Deque[Tuple[str, float]] = field(default_factory=lambda: deque(maxlen=15))
    name: str = "识别中"
    sim: float = 0.0
    feat_fresh: bool = False               # 本帧是否更新了特征（关键帧策略用）
    trail: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=40))
    feats_seen: int = 0

    @property
    def center(self):
        return (float(self.bbox[0] + self.bbox[2] / 2), float(self.bbox[1] + self.bbox[3] / 2))

    def predict(self):
        """匀速预测下一帧位置（用于 IoU 关联时补位）。"""
        x, y, w, h = self.bbox
        self.bbox = np.array([x + self.vel[0], y + self.vel[1], w, h], np.float32)
        return self.bbox

    def update(self, bbox, feat=None, score=1.0, alpha: float = 0.6):
        old_c = self.center
        x, y, w, h = bbox
        new_c = (x + w / 2, y + h / 2)
        v = np.array([new_c[0] - old_c[0], new_c[1] - old_c[1]], np.float32)
        self.vel = 0.5 * self.vel + 0.5 * v if self.age > 1 else v
        self.bbox = np.array(bbox, np.float32)
        self.score = float(score)
        self.hits += 1
        self.lost = 0
        self.trail.append((int(new_c[0]), int(new_c[1])))
        if feat is not None:
            f = np.asarray(feat, np.float32).ravel()
            if self.feat is None:
                self.feat = l2norm(f)
            else:
                self.feat = l2norm(alpha * l2norm(f) + (1 - alpha) * self.feat)
            self.feats_seen += 1
            self.feat_fresh = True

    def decide(self, min_hits: int = 3, vote_window: int = 15):
        """用时序投票决定显示身份。"""
        if self.feats_seen < min_hits or not self.votes:
            self.name = "识别中"
            return self.name
        recent = list(self.votes)[-vote_window:]
        if len(recent) == 1:            # 不做投票（窗口=1）时直接用当前帧结果
            self.name, self.sim = recent[0][0], recent[0][1]
            return self.name
        names = [n for n, _ in recent]
        cnt = Counter(names).most_common()
        top_name, top_cnt = cnt[0]
        # 需要多数票，且该身份的相似度中位数要过关
        if top_cnt < max(2, len(recent) // 2):
            self.name = self.name if self.name not in ("", "识别中") else "识别中"
            return self.name
        sims = np.array([s for n, s in recent if n == top_name], np.float32)
        self.name = top_name
        self.sim = float(np.median(sims))
        return self.name


class TrackManager:
    def __init__(self, iou_thr: float = 0.3, max_lost: int = 30,
                 ema_alpha: float = 0.6, min_hits: int = 3, vote_window: int = 15,
                 gating_dist: float = 0.5):
        self.iou_thr = float(iou_thr)
        self.max_lost = int(max_lost)
        self.ema_alpha = float(ema_alpha)
        self.min_hits = int(min_hits)
        self.vote_window = int(vote_window)
        self.gating_dist = float(gating_dist)     # 中心距离 / 框尺寸 的门限
        self.tracks: List[Track] = []
        self._next_id = 0
        self.frame_idx = 0

    # ---------------------------------------------------------------- step
    def step(self, faces: List, feats: Optional[np.ndarray], gallery: Gallery,
             threshold: float | None = None) -> List[Track]:
        """faces: List[Face]；feats: (N, D) 与 faces 对齐的特征；返回活跃轨迹。"""
        self.frame_idx += 1
        for t in self.tracks:
            t.age += 1
            t.feat_fresh = False

        bboxes = [f.bbox for f in faces] if faces else []

        # ---- 1) 关联：贪心匹配 (轨迹, 检测)
        pairs = []
        for ti, t in enumerate(self.tracks):
            for di, b in enumerate(bboxes):
                iou = bbox_iou(t.bbox, b)
                if iou >= self.iou_thr:
                    pairs.append((iou, ti, di))
                else:
                    # IoU 失效（快速运动）时用归一化中心距兜底
                    d = center_dist(t.bbox, b) / (0.5 * (t.bbox[2] + t.bbox[3]) + 1e-6)
                    if d < self.gating_dist:
                        pairs.append((0.5 * (1 - d) + 0.01, ti, di))
        pairs.sort(key=lambda x: -x[0])
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            f = faces[di]
            feat = feats[di] if (feats is not None and di < len(feats)) else None
            self.tracks[ti].update(f.bbox, feat, f.score, self.ema_alpha)

        # ---- 2) 未匹配的检测 → 新建轨迹
        for di, f in enumerate(faces):
            if di in used_d:
                continue
            t = Track(tid=self._next_id, bbox=f.bbox, score=f.score)
            self._next_id += 1
            feat = feats[di] if (feats is not None and di < len(feats)) else None
            if feat is not None:
                t.feat = l2norm(np.asarray(feat, np.float32))
                t.feats_seen = 1
            t.trail.append((int(t.center[0]), int(t.center[1])))
            self.tracks.append(t)

        # ---- 3) 未匹配的轨迹 → 计丢失
        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.lost += 1
                t.predict()

        # ---- 4) 识别 + 投票
        for t in self.tracks:
            if t.feat is not None and t.lost == 0 and t.feat_fresh:
                name, sim, _ = gallery.identify(t.feat, threshold=threshold)
                t.votes.append((name, sim))
                t.decide(self.min_hits, self.vote_window)

        # ---- 5) 清理
        self.tracks = [t for t in self.tracks if t.lost <= self.max_lost]
        return [t for t in self.tracks if t.lost == 0]

    def reset(self):
        self.tracks.clear()
        self._next_id = 0
        self.frame_idx = 0

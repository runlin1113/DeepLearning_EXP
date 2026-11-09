"""检测器公共数据结构与基类。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class Face:
    """一次人脸检测的结果。

    Attributes
    ----------
    bbox : (4,) ndarray  [x, y, w, h]  左上角 + 宽高（原图像素坐标）
    score : float        置信度
    landmarks : (5,2) ndarray 或 None
        5 个关键点。**顺序约定（本工程统一）**：
        [左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]（图像坐标系下的左右，
        与 ArcFace 112x112 参考模板一致）
    """
    bbox: np.ndarray
    score: float = 1.0
    landmarks: np.ndarray | None = None
    extra: dict = field(default_factory=dict)

    @property
    def x(self) -> float:
        return float(self.bbox[0])

    @property
    def y(self) -> float:
        return float(self.bbox[1])

    @property
    def w(self) -> float:
        return float(self.bbox[2])

    @property
    def h(self) -> float:
        return float(self.bbox[3])

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self):
        return self.x + self.w / 2, self.y + self.h / 2

    def crop(self, img: np.ndarray, pad: float = 0.0) -> np.ndarray:
        """按框裁剪（可外扩 pad 比例），越界自动夹取。"""
        H, W = img.shape[:2]
        px, py = self.w * pad, self.h * pad
        x1 = int(max(0, self.x - px))
        y1 = int(max(0, self.y - py))
        x2 = int(min(W, self.x + self.w + px))
        y2 = int(min(H, self.y + self.h + py))
        return img[y1:y2, x1:x2].copy()

    def clipped(self, W: int, H: int) -> "Face":
        x1 = max(0, self.x)
        y1 = max(0, self.y)
        x2 = min(W, self.x + self.w)
        y2 = min(H, self.y + self.h)
        lm = self.landmarks
        if lm is not None:
            lm = np.clip(lm, [[0, 0]], [[W - 1, H - 1]])
        return Face(np.array([x1, y1, max(0, x2 - x1), max(0, y2 - y1)], np.float32),
                    self.score, lm, self.extra)


class BaseDetector:
    """检测器接口：新增检测器只需实现 detect()。"""

    name = "base"

    def detect(self, img: np.ndarray, **kwargs) -> List[Face]:
        raise NotImplementedError

    def detect_one(self, img: np.ndarray) -> Face | None:
        """返回置信度最高的人脸（用于单人场景）。"""
        faces = self.detect(img)
        return max(faces, key=lambda f: f.score) if faces else None

    def __call__(self, img, **kwargs):
        return self.detect(img, **kwargs)


def nms_faces(faces: List[Face], iou_thr: float = 0.4) -> List[Face]:
    """通用 NMS（部分检测器导出已带 NMS，此处做兜底）。"""
    from ..utils import bbox_iou
    if not faces:
        return []
    faces = sorted(faces, key=lambda f: f.score, reverse=True)
    keep: List[Face] = []
    for f in faces:
        if all(bbox_iou(f.bbox, k.bbox) < iou_thr for k in keep):
            keep.append(f)
    return keep

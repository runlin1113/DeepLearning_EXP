"""
YuNet 人脸检测器（OpenCV 官方轻量模型，仅 230KB，自带 5 关键点）。

模型来源：https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
输出 N x 15：
    [x, y, w, h, re_x, re_y, le_x, le_y, nt_x, nt_y, rcm_x, rcm_y, lcm_x, lcm_y, score]
    re/le = 主体右眼/左眼（图像上看是"左/右"），nt = 鼻尖，rcm/lcm = 嘴角
    语义上 pts[0] 是**主体右眼**，在图像中位于左侧，正好对应 ArcFace 112x112 模板的第 0 点。

本类提供两种执行后端（backend='auto' 自动选择）：
    'cv2' : cv2.FaceDetectorYN  —— OpenCV 4.x 上最省事
    'ort' : onnxruntime + 手写解码 —— OpenCV 5.x 的新 ONNX 解析器读不了该模型时兜底，
            同时把 anchor-free 解码过程完全暴露出来，便于理解检测头的输出格式。

ONNX 原始输出（3 个 stride 8/16/32，热图分辨率 = 输入/stride，**anchor-free，无先验框**）：
    cls_{s}  (1, HW/s², 1)   分类分
    obj_{s}  (1, HW/s², 1)   目标性分     score = sqrt(cls * obj)
    bbox_{s} (1, HW/s², 4)   [dx, dy, log(w), log(h)]
    kps_{s}  (1, HW/s², 10)  5 点相对格点的偏移
解码（OpenCV 实现语义）：
    cx = (c + dx) * s,  cy = (r + dy) * s,  w = exp(dw) * s,  h = exp(dh) * s
    kpt_i = ((c + kx_i) * s, (r + ky_i) * s)
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from .base import BaseDetector, Face

_STRIDES = (8, 16, 32)


class YuNetDetector(BaseDetector):
    name = "yunet"

    def __init__(self,
                 model_path: str,
                 input_size=(640, 640),
                 score_threshold: float = 0.6,
                 nms_threshold: float = 0.3,
                 top_k: int = 5000,
                 max_side: int = 1280,
                 min_face: int = 24,
                 backend: str = "auto"):
        self.model_path = str(model_path)
        self.input_size = tuple(int(v) for v in input_size)      # (w, h) 模型输入
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.max_side = int(max_side)
        self.min_face = int(min_face)

        self.backend = None
        self._det = None
        self._sess = None
        self._cur_size = None

        if backend in ("auto", "cv2"):
            try:
                self._det = cv2.FaceDetectorYN.create(
                    self.model_path, "", self.input_size,
                    self.score_threshold, self.nms_threshold, self.top_k)
                self.backend = "cv2"
            except Exception as e:
                if backend == "cv2":
                    raise
                print(f"[YuNet] cv2.FaceDetectorYN 不可用（{str(e).splitlines()[0][:80]}），"
                      f"回退到 onnxruntime 后端")
        if self.backend is None:
            import onnxruntime as ort
            from ..config import resolve_onnx_providers
            self._sess = ort.InferenceSession(self.model_path,
                                              providers=resolve_onnx_providers(False))
            self._in_name = self._sess.get_inputs()[0].name
            self._out_names = [o.name for o in self._sess.get_outputs()]
            shp = self._sess.get_inputs()[0].shape
            if isinstance(shp[2], int):
                self.input_size = (int(shp[3]), int(shp[2]))
            self.backend = "ort"

    # ================================================================ 入口
    def detect(self, img: np.ndarray, max_side: int | None = None, **kwargs) -> List[Face]:
        if img is None or img.size == 0:
            return []
        limit = self.max_side if max_side is None else int(max_side)
        H, W = img.shape[:2]
        scale = 1.0
        work = img
        if limit and max(W, H) > limit:          # 大图降采样：精度几乎无损，速度线性提升
            scale = limit / float(max(W, H))
            work = cv2.resize(img, (int(round(W * scale)), int(round(H * scale))))
        faces = (self._detect_cv2 if self.backend == "cv2" else self._detect_ort)(work)
        inv = 1.0 / scale
        out: List[Face] = []
        for f in faces:
            if f.score < self.score_threshold:
                continue
            if min(f.w, f.h) * inv < self.min_face:
                continue
            if scale != 1.0:
                f.bbox = f.bbox * inv
                if f.landmarks is not None:
                    f.landmarks = f.landmarks * inv
            out.append(f)
        out.sort(key=lambda f: f.score, reverse=True)
        return out

    # ------------------------------------------------------------ cv2 后端
    def _detect_cv2(self, work):
        h, w = work.shape[:2]
        if self._cur_size != (w, h):
            self._det.setInputSize((w, h))
            self._cur_size = (w, h)
        _, raw = self._det.detect(work)
        if raw is None:
            return []
        faces = []
        for row in raw:
            pts = row[4:14].reshape(5, 2).astype(np.float32)
            faces.append(Face(np.array(row[0:4], np.float32), float(row[14]), pts))
        return faces

    # ------------------------------------------------------------ ort 后端
    def _preprocess(self, work):
        """letterbox 到模型输入尺寸（保持长宽比，右下补 0）。"""
        in_w, in_h = self.input_size
        h, w = work.shape[:2]
        s = min(in_w / w, in_h / h)
        nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
        canvas = np.zeros((in_h, in_w, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(work, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)  # RGB, 0-255
        return blob, s

    def _detect_ort(self, work):
        blob, s = self._preprocess(work)
        outs = dict(zip(self._out_names, self._sess.run(self._out_names, {self._in_name: blob})))
        in_w, in_h = self.input_size

        boxes, scores, all_pts = [], [], []
        for st in _STRIDES:
            cls = outs[f"cls_{st}"][0].reshape(-1)
            obj = outs[f"obj_{st}"][0].reshape(-1)
            bbox = outs[f"bbox_{st}"][0].reshape(-1, 4)
            kps = outs[f"kps_{st}"][0].reshape(-1, 10)
            score = np.sqrt(np.clip(cls, 0, 1) * np.clip(obj, 0, 1))
            keep = score >= min(self.score_threshold, 0.05)
            if not keep.any():
                continue
            hw = in_h // st, in_w // st
            idx = np.nonzero(keep)[0]
            rr, cc = idx // hw[1], idx % hw[1]
            b = bbox[idx]
            cx = (cc + b[:, 0]) * st
            cy = (rr + b[:, 1]) * st
            bw = np.exp(np.clip(b[:, 2], -6, 6)) * st
            bh = np.exp(np.clip(b[:, 3], -6, 6)) * st
            k = kps[idx].reshape(-1, 5, 2).copy()
            k[:, :, 0] = (cc[:, None] + k[:, :, 0]) * st
            k[:, :, 1] = (rr[:, None] + k[:, :, 1]) * st
            boxes.append(np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1))
            scores.append(score[idx])
            all_pts.append(k)

        if not boxes:
            return []
        boxes = np.concatenate(boxes).astype(np.float32)
        scores = np.concatenate(scores).astype(np.float32)
        all_pts = np.concatenate(all_pts).astype(np.float32)

        # 还原到 work 坐标
        boxes[:, :4:2] /= s
        boxes[:, 1:4:2] /= s
        all_pts /= s

        # 裁到图像内
        H, W = work.shape[:2]
        boxes[:, 0] = np.clip(boxes[:, 0], 0, W - 1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, H - 1)
        boxes[:, 2] = np.clip(boxes[:, 2], 1, W - boxes[:, 0])
        boxes[:, 3] = np.clip(boxes[:, 3], 1, H - boxes[:, 1])

        idx = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(),
                               self.score_threshold, self.nms_threshold)
        if len(idx) == 0:
            return []
        idx = np.asarray(idx).reshape(-1)
        return [Face(boxes[i], float(scores[i]), all_pts[i]) for i in idx[: self.top_k]]

    # ------------------------------------------------------------- 参数调节
    def set_score_threshold(self, v: float):
        self.score_threshold = float(v)
        if self.backend == "cv2":
            self._det.setScoreThreshold(float(v))

    def set_nms_threshold(self, v: float):
        self.nms_threshold = float(v)
        if self.backend == "cv2":
            self._det.setNMSThreshold(float(v))

    def __repr__(self):
        return (f"YuNetDetector(model={Path(self.model_path).name}, backend={self.backend}, "
                f"input={self.input_size}, score>{self.score_threshold})")

"""
CenterFace 人脸检测器（可选后端，CenterNet 风格 anchor-free）。

论文：CenterFace: Joint Face Detection and Alignment Using Face as Point (2020)
权重：models/centerface.onnx（约 7MB），由 scripts/step0_download_models.py 尝试自动下载。
输出 4 个张量（步长 4），热图分辨率即输入尺寸的 1/4：
    heatmap  (1, 1, H/4, W/4)
    scale    (1, 2, H/4, W/4)   # 宽高
    offset   (1, 2, H/4, W/4)   # 中心偏移
    landmarks(1,10, H/4, W/4)   # 5 个点 (x,y) x 5

本工程把它与 YuNet 统一成 BaseDetector 接口，因此对齐/识别模块对检测器完全解耦
——这就是"替换原生检测器"能一行切换的原因。
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from .base import BaseDetector, Face


class CenterFaceDetector(BaseDetector):
    name = "centerface"

    def __init__(self, model_path: str, score_threshold: float = 0.5,
                 nms_threshold: float = 0.3, max_side: int = 1280,
                 min_face: int = 24, prefer_gpu: bool = False):
        import onnxruntime as ort
        from ..config import resolve_onnx_providers

        model_path = str(model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"未找到 CenterFace 权重: {model_path}\n"
                f"请先运行 scripts/step0_download_models.py，或改用 YuNet（默认）。")
        self.sess = ort.InferenceSession(model_path, providers=resolve_onnx_providers(prefer_gpu))
        self.input_name = self.sess.get_inputs()[0].name
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.max_side = int(max_side)
        self.min_face = int(min_face)

    # --------------------------------------------------------------- core
    def detect(self, img: np.ndarray, max_side: int | None = None, **kwargs) -> List[Face]:
        if img is None or img.size == 0:
            return []
        H, W = img.shape[:2]
        limit = self.max_side if max_side is None else int(max_side)
        scale = 1.0
        work = img
        if limit and max(W, H) > limit:
            scale = limit / float(max(W, H))
            work = cv2.resize(img, (int(W * scale), int(H * scale)))

        # 输入尺寸对齐到 32 的倍数（2^5 下采样）
        h, w = work.shape[:2]
        w32, h32 = max(32, w // 32 * 32), max(32, h // 32 * 32)
        inp = cv2.resize(work, (w32, h32))
        blob = cv2.dnn.blobFromImage(inp, 1.0, (w32, h32), (0, 0, 0),
                                     swapRB=True, crop=False)
        outs = self.sess.run(None, {self.input_name: blob})

        heatmap = outs[0][0, 0]
        scale_hw = outs[0][1]
        offset = outs[1][0] if outs[1].ndim == 4 else outs[1]
        lms = outs[2][0] if outs[2].ndim == 4 else outs[2]

        ys, xs = np.where(heatmap > self.score_threshold)
        if len(ys) == 0:
            return []
        scores = heatmap[ys, xs]

        # 关键：heatmap 是 1/4 分辨率，坐标要乘 4 再加中心偏移
        s0 = scale_hw[0, ys, xs]
        s1 = scale_hw[1, ys, xs]
        ox = offset[0, ys, xs]
        oy = offset[1, ys, xs]
        cx = (xs + ox) * 4.0
        cy = (ys + oy) * 4.0

        lm = lms[:, ys, xs].reshape(5, 2, -1).transpose(2, 0, 1)  # (N,5,2)
        lm = lm.copy()
        lm[:, :, 0] = lm[:, :, 0] * 4.0 + (xs[:, None] * 4.0)
        lm[:, :, 1] = lm[:, :, 1] * 4.0 + (ys[:, None] * 4.0)

        faces: List[Face] = []
        inv = 1.0 / scale
        # 还原到 work 尺度（w32/h32）再映射回原图
        kx, ky = w / float(w32), h / float(h32)
        for i in range(len(xs)):
            bw, bh = float(s0[i]), float(s1[i])
            if min(bw, bh) * inv < self.min_face:
                continue
            x = (cx[i] - bw / 2) * kx * inv
            y = (cy[i] - bh / 2) * ky * inv
            pts = lm[i].astype(np.float32) * np.array([kx * inv, ky * inv], np.float32)
            faces.append(Face(np.array([x, y, bw * kx * inv, bh * ky * inv], np.float32),
                              float(scores[i]), pts))

        # 局部极大值 + NMS
        faces.sort(key=lambda f: f.score, reverse=True)
        faces = self._local_max(faces)
        return faces

    @staticmethod
    def _local_max(faces: List[Face], iou_thr: float = 0.3) -> List[Face]:
        from ..utils import bbox_iou
        keep: List[Face] = []
        for f in faces:
            if all(bbox_iou(f.bbox, k.bbox) < iou_thr for k in keep):
                keep.append(f)
        return keep

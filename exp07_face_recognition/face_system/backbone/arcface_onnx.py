"""ArcFace 特征提取（推理侧，ONNX Runtime）。

模型：w600k_r50.onnx  ——  ArcFace + ResNet50 骨干，Glint360K(36 万人) 训练，输出 512 维。
预处理（与 insightface 官方实现完全一致）：
    BGR->RGB, x = (x - 127.5) / 127.5, NCHW
输出做 L2 归一化后，内积即余弦相似度。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

from ..config import resolve_onnx_providers
from ..utils import l2norm


class ArcFaceONNX:
    def __init__(self, model_path: str, prefer_gpu: bool = False, intra_threads: int = 0):
        import onnxruntime as ort
        model_path = str(model_path)
        if not Path(model_path).exists():
            raise FileNotFoundError(f"未找到 ArcFace 权重: {model_path}")

        so = ort.SessionOptions()
        so.log_severity_level = 3          # 屏蔽 "output shape {1,512} vs {N,512}" 之类告警
        if intra_threads > 0:
            so.intra_op_num_threads = intra_threads
        self.sess = ort.InferenceSession(model_path, sess_options=so,
                                         providers=resolve_onnx_providers(prefer_gpu))
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        shape = self.sess.get_inputs()[0].shape
        self.input_hw = (int(shape[2]) if isinstance(shape[2], int) else 112,
                         int(shape[3]) if isinstance(shape[3], int) else 112)
        self.provider = self.sess.get_providers()[0]
        self._n_call = 0
        self._total_ms = 0.0

    # ------------------------------------------------------------ preproc
    def _preprocess(self, faces: List[np.ndarray]) -> np.ndarray:
        h, w = self.input_hw
        batch = np.empty((len(faces), 3, h, w), np.float32)
        for i, im in enumerate(faces):
            if im is None:
                raise ValueError("输入人脸为空")
            if im.shape[0] != h or im.shape[1] != w:
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)
            rgb = (rgb - 127.5) / 127.5
            batch[i] = rgb.transpose(2, 0, 1)
        return batch

    # ---------------------------------------------------------------- feat
    def get_feat(self, faces: Iterable[np.ndarray], normalize: bool = True,
                 batch_size: int = 8) -> np.ndarray:
        """faces: 若干张 112x112 BGR 对齐人脸 -> (N, 512) 特征"""
        faces = [f for f in faces if f is not None]
        if not faces:
            return np.zeros((0, 512), np.float32)
        t0 = time.perf_counter()
        feats = []
        for i in range(0, len(faces), batch_size):
            chunk = faces[i:i + batch_size]
            blob = self._preprocess(chunk)
            out = self.sess.run(self.output_names, {self.input_name: blob})[0]
            feats.append(np.asarray(out, np.float32).reshape(len(chunk), -1))
        out = np.vstack(feats)
        self._n_call += len(faces)
        self._total_ms += (time.perf_counter() - t0) * 1000
        return l2norm(out) if normalize else out

    def get_feat_one(self, face: np.ndarray) -> np.ndarray:
        return self.get_feat([face])[0]

    @property
    def avg_ms(self) -> float:
        return self._total_ms / max(1, self._n_call)

    def __repr__(self):
        return (f"ArcFaceONNX({Path(self.sess._model_path).name if hasattr(self.sess, '_model_path') else 'w600k_r50'}, "
                f"dim=512, provider={self.provider})")

"""
人脸检测与动态识别系统（课程实验七）
=====================================
处理链路：
    图像/视频 → 检测(YuNet/CenterFace) → 对齐(face.evoLVe 改造) → ArcFace 特征
             → 余弦比对(1:1 / 1:N) → 视频追踪 + 多帧融合 → 输出

快速开始：
    from face_system import build_pipeline
    det, aligner, rec = build_pipeline()
    for af in aligner.run(img):
        feat = rec.get_feat_one(af.crop)
"""
from __future__ import annotations

from .config import CFG, DetectorCfg, AlignCfg, RecCfg, setup_cuda_env
from .detectors.base import BaseDetector, Face
from .detectors.yunet import YuNetDetector
from .align.align_pipeline import FaceAligner, AlignedFace, align_folder
from .backbone.arcface_onnx import ArcFaceONNX
from .gallery import Gallery
from .tracker import Track, TrackManager

__all__ = [
    "CFG", "DetectorCfg", "AlignCfg", "RecCfg",
    "BaseDetector", "Face", "YuNetDetector",
    "FaceAligner", "AlignedFace", "align_folder",
    "ArcFaceONNX", "Gallery", "Track", "TrackManager",
    "build_detector", "build_pipeline",
]


def build_detector(name: str | None = None, prefer_gpu: bool = False, **kwargs) -> BaseDetector:
    """检测器工厂：'yunet' | 'centerface'（新增检测器只需在这里注册）"""
    setup_cuda_env()
    name = (name or CFG.detector.name).lower()
    d = CFG.detector
    if name == "yunet":
        return YuNetDetector(
            model_path=d.yunet_model,
            input_size=d.input_size,
            score_threshold=kwargs.get("score_threshold", d.score_threshold),
            nms_threshold=kwargs.get("nms_threshold", d.nms_threshold),
            top_k=d.top_k, max_side=d.max_side, min_face=d.min_face,
        )
    if name == "centerface":
        from .detectors.centerface import CenterFaceDetector
        return CenterFaceDetector(
            model_path=d.centerface_model,
            score_threshold=kwargs.get("score_threshold", 0.5),
            nms_threshold=kwargs.get("nms_threshold", d.nms_threshold),
            max_side=d.max_side, min_face=d.min_face, prefer_gpu=prefer_gpu,
        )
    raise ValueError(f"未知检测器: {name}（可选 yunet / centerface）")


def build_pipeline(detector: str | None = None, prefer_gpu: bool = False,
                   crop_size: int | None = None):
    """一把梭：返回 (detector, aligner, rec)"""
    det = build_detector(detector, prefer_gpu=prefer_gpu)
    aligner = FaceAligner(det, crop_size=crop_size or CFG.align.crop_size,
                          align_type=CFG.align.align_type,
                          auto_fix_reflection=CFG.align.auto_fix_reflection)
    rec = ArcFaceONNX(CFG.rec.model, prefer_gpu=prefer_gpu)
    return det, aligner, rec

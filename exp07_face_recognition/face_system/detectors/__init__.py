"""人脸检测器集合：YuNet（默认）/ CenterFace。"""
from .base import BaseDetector, Face, nms_faces
from .yunet import YuNetDetector

__all__ = ["BaseDetector", "Face", "nms_faces", "YuNetDetector"]

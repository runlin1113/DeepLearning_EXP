"""特征提取骨干：ArcFace(ONNX 推理) / iResNet(PyTorch 训练)。"""
from .arcface_onnx import ArcFaceONNX

__all__ = ["ArcFaceONNX"]

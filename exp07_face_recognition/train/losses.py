"""
损失函数：Softmax / CosFace / ArcFace
=====================================
三者共用同一个"归一化特征 + 归一化权重"的判别头，区别只在 logit 上：

    logits_j = s · cos(θ_j)                    (Softmax, 无 margin)
    logits_y = s · (cos(θ_y) − m)              (CosFace, 余弦 margin)
    logits_y = s · cos(θ_y + m)                (ArcFace, 角度 margin)
    logits_j = s · cos(θ_j),  j ≠ y

符号：f = 特征（‖f‖=1），W = 分类器权重（‖W_j‖=1），θ_y 是 f 与正确类中心的夹角。
s 控制 softmax 的"温度/锐度"（超球半径），m 强制"类内更紧、类间更开"。

ArcFace 的目标 logit 需要由 cos(θ) 反解 θ 再加 m，用恒等式：
    cos(θ + m) = cosθ·cos m − sinθ·sin m,  sinθ = sqrt(1 − cos²θ)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginHead(nn.Module):
    """归一化特征 + 归一化权重 → 加性/余弦 margin 的 logits。

    mode: 'softmax' | 'cosface' | 'arcface'
    """

    def __init__(self, emb_dim: int, num_classes: int, mode: str = "arcface",
                 scale: float = 30.0, margin: float = 0.35, easy_margin: bool = False):
        super().__init__()
        assert mode in ("softmax", "cosface", "arcface")
        self.mode = mode
        self.s = float(scale)
        self.m = float(margin)
        self.easy_margin = easy_margin
        self.weight = nn.Parameter(torch.randn(num_classes, emb_dim) * 0.01)
        nn.init.xavier_uniform_(self.weight)

    def cosine_logits(self, feat: torch.Tensor):
        """不加 margin 的纯净余弦 logits（用于计算可解释的分类准确率）"""
        cos = F.linear(F.normalize(feat), F.normalize(self.weight))
        return cos * self.s

    def forward(self, feat: torch.Tensor, label: torch.Tensor):
        cos = F.linear(F.normalize(feat), F.normalize(self.weight)).clamp(-1 + 1e-7, 1 - 1e-7)

        if self.mode == "softmax" or self.m == 0:
            return cos * self.s

        if self.mode == "cosface":
            target = cos - self.m
        else:  # arcface
            theta = torch.acos(cos)
            target = torch.cos(theta + self.m)
            if self.easy_margin:
                target = torch.where(cos > 0, target, cos)

        onehot = torch.zeros_like(cos)
        onehot.scatter_(1, label.view(-1, 1), 1.0)
        logits = onehot * target + (1 - onehot) * cos
        return logits * self.s


def build_head(mode: str, emb_dim: int, num_classes: int, s: float, m: float):
    return MarginHead(emb_dim, num_classes, mode=mode, scale=s, margin=m)


# ------------------------------------------------------------------ 参考实现
class ArcFaceLoss(nn.Module):
    """独立可复用的 ArcFace 损失（等价于 MarginHead(mode='arcface') + CrossEntropy）"""

    def __init__(self, emb_dim=512, num_classes=10572, s=64.0, m=0.5):
        super().__init__()
        self.head = MarginHead(emb_dim, num_classes, "arcface", s, m)
        self.ce = nn.CrossEntropyLoss()

    def forward(self, feat, label):
        return self.ce(self.head(feat, label), label)


def cosine_margin_to_degrees(m: float, s: float) -> float:
    """把常用的 (s=64, m=0.5) 换算成直观的"角度间隔"：m 是弧度，0.5 rad ≈ 28.6°。"""
    return m * 180 / math.pi

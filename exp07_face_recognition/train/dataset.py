"""
训练数据集：读取"按身份分文件夹"的**已对齐** 112x112 人脸。

目录格式（本工程 step2 的产物）：
    data/lfw20_aligned/train/<身份名>/*.jpg
    data/lfw20_aligned/valid/<身份名>/*.jpg

训练增强（人脸识别通用且必要的几项）：
    - 随机水平翻转（人脸识别中要谨慎：翻转会改变左右脸语义，
      这里按 p=0.5 使用，属于常规做法，因为在验证集上也一致地"不翻转"）
    - 随机仿射：旋转 ±10°、缩放 0.9~1.1、平移 ±5%（模拟姿态/构图抖动）
    - 亮度/对比度/饱和度轻扰动
    - 随机擦除（Random Erasing，模拟遮挡，提升鲁棒性）
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from face_system.utils import imread, list_images


class AlignedFaceDataset(Dataset):
    def __init__(self, root, image_size: int = 112, train: bool = True,
                 classes=None, max_per_class: int = 0):
        self.root = Path(root)
        self.image_size = image_size
        self.train = train

        dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        if classes is not None:
            keep = set(classes)
            dirs = [d for d in dirs if d.name in keep]

        self.classes = [d.name for d in dirs]
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for d in dirs:
            imgs = list_images(d)
            if max_per_class:
                imgs = imgs[:max_per_class]
            for p in imgs:
                self.samples.append((p, self.class_to_idx[d.name]))
        if not self.samples:
            raise RuntimeError(f"{root} 下没有找到图片，请先运行 scripts/step2_detect_align.py")

    def __len__(self):
        return len(self.samples)

    def _aug(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if random.random() < 0.5:                      # 水平翻转
            img = cv2.flip(img, 1)
        if random.random() < 0.7:                      # 随机仿射
            ang = random.uniform(-10, 10)
            sc = random.uniform(0.9, 1.1)
            tx = random.uniform(-0.05, 0.05) * w
            ty = random.uniform(-0.05, 0.05) * h
            M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
            M[0, 2] += tx
            M[1, 2] += ty
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        if random.random() < 0.5:                      # 光度扰动
            img = img.astype(np.float32)
            img = img * random.uniform(0.85, 1.15) + random.uniform(-12, 12)
            img = np.clip(img, 0, 255).astype(np.uint8)
        if random.random() < 0.25:                     # 随机擦除
            eh = random.randint(h // 12, h // 4)
            ew = random.randint(w // 12, w // 4)
            y0 = random.randint(0, h - eh)
            x0 = random.randint(0, w - ew)
            img[y0:y0 + eh, x0:x0 + ew] = np.random.randint(0, 255, 3, np.uint8)
        return img

    def __getitem__(self, idx):
        p, label = self.samples[idx]
        img = imread(p)
        if img is None:
            img = np.zeros((self.image_size, self.image_size, 3), np.uint8)
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(img, (self.image_size, self.image_size))
        if self.train:
            img = self._aug(img)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 127.5
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())
        return tensor, label


def build_datasets(train_root, valid_root, image_size=112, max_per_class=0) -> Tuple[
        AlignedFaceDataset, AlignedFaceDataset]:
    tr = AlignedFaceDataset(train_root, image_size, train=True, max_per_class=max_per_class)
    va = AlignedFaceDataset(valid_root, image_size, train=False,
                            classes=tr.classes, max_per_class=max_per_class)
    return tr, va

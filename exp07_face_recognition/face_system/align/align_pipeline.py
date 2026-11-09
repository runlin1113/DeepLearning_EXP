"""
★ 本实验的核心改造：把 face.evoLVe 的"对齐"和 YuNet/CenterFace 的"检测"接在一起。

原版链路（face.evoLVe/application/align/face_align.py）：
    PIL 读图 → MTCNN(detect_faces) 出 bbox + 5 点 → warp_and_crop_face → 保存对齐图

改造后链路（本文件）：
    任意 BaseDetector（YuNet / CenterFace）→ Face(bbox, 5 landmarks)
        → 关键点顺序归一化 + 镜像自纠正
        → align_trans.warp_and_crop_face(相似变换) → 112x112

设计要点
--------
1. 检测与对齐解耦：对齐模块只认 `Face`，换检测器不改对齐代码；
2. 关键点顺序：ArcFace 模板为 [图像左眼, 右眼, 鼻尖, 左嘴角, 右嘴角]，
   YuNet 输出的第 0 点是"主体右眼"（图像左侧），语义一致，无需交换；
   但不同检测器语义不同（有的按主体、有的按图像），因此加了**镜像自纠正**：
   求解相似变换后检查 2x3 矩阵的 2x2 子块行列式，若为负说明解是"镜像解"，
   交换左右眼+左右嘴角重解一次，取行列式为正（保向）的解。
3. 相似变换 vs 仿射：人脸对齐要保证不拉伸五官比例，所以必须用
   similarity（4 自由度），不能直接用 bbox 拉伸 resize —— 后者会引入
   尺度/长宽比畸变，直接拉低余弦相似度（见 step6 的对齐消融实验）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence

import numpy as np

from ..detectors.base import BaseDetector, Face
from ..utils import imread, imwrite, list_images
from .align_trans import get_reference_facial_points, warp_and_crop_face

ARCFACE_REF_112 = get_reference_facial_points((112, 112), 0, (0, 0), default_square=True)

# 只有左右眼/左右嘴角会因"主体/图像"语义不同而互换
_MIRROR_SWAP = [1, 0, 2, 4, 3]


@dataclass
class AlignedFace:
    """一张对齐好的人脸 + 全部中间结果（便于可视化与调试）"""
    crop: np.ndarray                      # 112x112 BGR
    face: Face                            # 原图上的框与 5 点
    tfm: np.ndarray                       # 2x3 相似变换矩阵
    swapped: bool = False                 # 是否触发了左右互换
    det_sign: float = 1.0                 # 变换矩阵 2x2 行列式符号
    eye_dy: float = 0.0                   # 对齐后双眼 y 差（理想≈0）
    extra: dict = field(default_factory=dict)


class FaceAligner:
    """检测 → 5 点 → 相似变换 → 112x112 标准人脸"""

    def __init__(self,
                 detector: BaseDetector | None = None,
                 crop_size: int = 112,
                 align_type: str = "similarity",
                 auto_fix_reflection: bool = True,
                 reference_pts: np.ndarray | None = None):
        self.detector = detector
        self.crop_size = int(crop_size)
        self.align_type = align_type
        self.auto_fix_reflection = auto_fix_reflection

        if reference_pts is None:
            if self.crop_size == 112:
                reference_pts = ARCFACE_REF_112
            else:
                scale = self.crop_size / 112.0
                reference_pts = ARCFACE_REF_112 * scale
        self.ref_pts = np.asarray(reference_pts, np.float32).reshape(5, 2)

    # ------------------------------------------------------------ 变换求解
    def _solve_tfm(self, pts5: np.ndarray):
        """求解 2x3 变换；返回 (tfm, swapped, det_sign)"""
        from .matlab_cp2tform import get_similarity_transform_for_cv2
        pts5 = np.asarray(pts5, np.float32).reshape(5, 2)
        tfm = get_similarity_transform_for_cv2(pts5, self.ref_pts)
        det = float(np.linalg.det(tfm[:, :2]))

        if det < 0 and self.auto_fix_reflection:
            pts2 = pts5[_MIRROR_SWAP]
            tfm2 = get_similarity_transform_for_cv2(pts2, self.ref_pts)
            det2 = float(np.linalg.det(tfm2[:, :2]))
            if det2 > 0:
                return tfm2, True, det2
        return tfm, False, det

    # ---------------------------------------------------------------- 对齐
    def align_face(self, img: np.ndarray, face: Face) -> AlignedFace:
        if face.landmarks is None:
            raise ValueError("该 Face 没有关键点，无法做相似变换对齐")
        tfm, swapped, det = self._solve_tfm(face.landmarks)
        crop = warp_and_crop_face(
            img,
            face.landmarks if not swapped else face.landmarks[_MIRROR_SWAP],
            reference_pts=self.ref_pts,
            crop_size=(self.crop_size, self.crop_size),
            align_type=self.align_type,
        )
        # 对齐后双眼水平度（越小越"正"）
        pts = np.hstack([face.landmarks, np.ones((5, 1))]) @ np.vstack([tfm, [0, 0, 1]]).T
        eye_dy = float(abs(pts[0, 1] - pts[1, 1]))
        return AlignedFace(crop=crop, face=face, tfm=tfm, swapped=swapped,
                           det_sign=det, eye_dy=eye_dy)

    def align_points(self, img: np.ndarray, pts5: np.ndarray) -> np.ndarray:
        """只有 5 点（例如来自标注文件）时也能对齐。"""
        crop = warp_and_crop_face(img, np.asarray(pts5, np.float32), self.ref_pts,
                                  (self.crop_size, self.crop_size), self.align_type)
        return crop

    # ------------------------------------------------------- 端到端：图 → 人脸
    def run(self, img: np.ndarray, max_faces: int = 0) -> List[AlignedFace]:
        """检测 + 对齐，返回全部人脸。max_faces>0 时只留最大的几张。"""
        if self.detector is None:
            raise RuntimeError("FaceAligner 未绑定 detector，无法端到端运行")
        faces = self.detector.detect(img)
        if max_faces > 0 and len(faces) > max_faces:
            faces = sorted(faces, key=lambda f: f.area, reverse=True)[:max_faces]
        out = []
        for f in faces:
            try:
                out.append(self.align_face(img, f))
            except Exception as e:      # 单张人脸失败不影响整体
                print(f"[warn] align failed: {e}")
        return out

    @staticmethod
    def load_and_run(path, aligner: "FaceAligner", max_faces: int = 0):
        img = imread(path)
        if img is None:
            raise FileNotFoundError(f"读不到图片: {path}")
        return img, aligner.run(img, max_faces=max_faces)


def align_folder(src_root, dst_root, aligner: FaceAligner,
                 crop_size: int = 112, limit_per_id: int = 0,
                 save_fail_log: Path | None = None):
    """
    批量对齐一个"按身份分文件夹"的数据集（替代原版 face_align.py 的 MTCNN 循环）。

    src_root/
        id_a/*.jpg
        id_b/*.jpg
    dst_root/
        id_a/*.jpg  (112x112)
    """
    src_root, dst_root = Path(src_root), Path(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)
    stats = {"total": 0, "ok": 0, "fail": 0, "multi": 0, "swapped": 0}
    fails: List[str] = []

    for sub in sorted([p for p in src_root.iterdir() if p.is_dir()]):
        out_sub = dst_root / sub.name
        out_sub.mkdir(parents=True, exist_ok=True)
        imgs = list_images(sub)
        if limit_per_id:
            imgs = imgs[:limit_per_id]
        for p in imgs:
            stats["total"] += 1
            img = imread(p)
            if img is None:
                stats["fail"] += 1
                fails.append(f"{p}\tunreadable")
                continue
            faces = aligner.detector.detect(img)
            if not faces:
                stats["fail"] += 1
                fails.append(f"{p}\tno_face")
                continue
            if len(faces) > 1:
                stats["multi"] += 1
            face = max(faces, key=lambda f: f.area)      # 数据集场景取最大人脸
            try:
                af = aligner.align_face(img, face)
            except Exception as e:
                stats["fail"] += 1
                fails.append(f"{p}\t{e}")
                continue
            stats["ok"] += 1
            stats["swapped"] += int(af.swapped)
            imwrite(out_sub / (p.stem + ".jpg"), af.crop)

    if save_fail_log:
        Path(save_fail_log).parent.mkdir(parents=True, exist_ok=True)
        Path(save_fail_log).write_text("\n".join(fails), encoding="utf-8")
    return stats

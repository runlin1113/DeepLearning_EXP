"""
[复用的第三方源码] face.evoLVe  ——  applications/align/align_trans.py
来源：https://github.com/ZhaoJ9014/face.evoLVe （MIT License）
作用：给出 ArcFace 标准的 5 点参考模板，并用相似变换把任意人脸 warp 到标准姿态。

================================================================
本实验对原码的改造点（对应"修改源码、替换原生检测器"的要求）
================================================================
1) 删除对 MTCNN 的隐式依赖：原版 `face_align.py` 里用 `detect_faces()` 出 5 点，
   这里把 5 点改成**从外部传入**（由 YuNet / CenterFace 提供），
   对齐函数本身与检测器完全解耦。
2) 修复 `warp_and_crop_face(crop_size=(112,112))` 且 reference_pts=None 时
   原版会抛 FaceWarpException 的问题：正方形输出时自动改用
   `get_reference_facial_points(default_square=True)`。
3) 修复 `if align_type is 'cv2_affine'`（Python 3.8+ 语法告警）→ 用 `==`。
4) 修复 `FaceWarpException.__str__` 中 `super.__str__(self)` 的非法写法。
5) 增加 `align_type` 缺省值与类型校验，避免拼写错误静默走 similarity 分支。

注：原版拼写 'smilarity'（少一个 i）被保留为兼容别名 —— 这是一个真实存在的坑：
    拼错时不会报错，而是**静默**退回 similarity 分支。
"""
from __future__ import annotations

import cv2
import numpy as np

from .matlab_cp2tform import get_similarity_transform_for_cv2

# ArcFace 官方 5 点模板（crop_size = 96x112 下的坐标）
REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],   # 左眼（图像左）
    [65.53179932, 51.50139999],   # 右眼
    [48.02519989, 71.73660278],   # 鼻尖
    [33.54930115, 92.36550140],   # 左嘴角
    [62.72990036, 92.20410156],   # 右嘴角
]

DEFAULT_CROP_SIZE = (96, 112)


class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(__file__, super().__str__())


def get_reference_facial_points(output_size=None,
                                inner_padding_factor=0.0,
                                outer_padding=(0, 0),
                                default_square=False):
    """根据裁剪参数生成 5 点参考坐标。

    `default_square=True` 且 output_size=(112,112) 时，返回的正是
    ArcFace 112x112 标准模板：
        [[38.29, 51.70], [73.53, 51.50], [56.03, 71.74],
         [41.55, 92.37], [70.73, 92.20]]
    （把 96x112 的模板整体右移 (112-96)/2 = 8 像素得到）
    """
    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS, dtype=np.float64)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE, dtype=np.float64)

    # 0) 把内部区域变成正方形（宽度补齐到 112）
    if default_square:
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff

    if (output_size is not None
            and output_size[0] == tmp_crop_size[0]
            and output_size[1] == tmp_crop_size[1]):
        return tmp_5pts

    if inner_padding_factor == 0 and tuple(outer_padding) == (0, 0):
        if output_size is None:
            return tmp_5pts
        else:
            raise FaceWarpException(
                'No paddings to do, output_size must be None or {}'.format(tmp_crop_size))

    if not (0 <= inner_padding_factor <= 1.0):
        raise FaceWarpException('Not (0 <= inner_padding_factor <= 1.0)')

    if ((inner_padding_factor > 0 or outer_padding[0] > 0 or outer_padding[1] > 0)
            and output_size is None):
        output_size = tmp_crop_size * (1 + inner_padding_factor * 2).astype(np.int32)
        output_size += np.array(outer_padding)

    if not (outer_padding[0] < output_size[0] and outer_padding[1] < output_size[1]):
        raise FaceWarpException('Not (outer_padding[0] < output_size[0]'
                                'and outer_padding[1] < output_size[1])')

    # 1) 按 inner_padding_factor 内扩
    if inner_padding_factor > 0:
        size_diff = tmp_crop_size * inner_padding_factor * 2
        tmp_5pts += size_diff / 2
        tmp_crop_size += np.round(size_diff).astype(np.int32)

    # 2) 缩放到 (output_size - outer_padding*2)
    size_bf_outer_pad = np.array(output_size) - np.array(outer_padding) * 2
    if size_bf_outer_pad[0] * tmp_crop_size[1] != size_bf_outer_pad[1] * tmp_crop_size[0]:
        raise FaceWarpException('Must have (output_size - outer_padding)'
                                '= some_scale * (crop_size * (1.0 + inner_padding_factor)')

    scale_factor = size_bf_outer_pad[0].astype(np.float32) / tmp_crop_size[0]
    tmp_5pts = tmp_5pts * scale_factor
    tmp_crop_size = size_bf_outer_pad

    # 3) 加外扩
    return tmp_5pts + np.array(outer_padding)


def get_affine_transform_matrix(src_pts, dst_pts):
    """最小二乘求 2x3 仿射矩阵（使用全部 K 个点，不约束为相似变换）。"""
    tfm = np.float32([[1, 0, 0], [0, 1, 0]])
    n_pts = src_pts.shape[0]
    ones = np.ones((n_pts, 1), src_pts.dtype)
    src_pts_ = np.hstack([src_pts, ones])
    dst_pts_ = np.hstack([dst_pts, ones])

    A, res, rk, s = np.linalg.lstsq(src_pts_, dst_pts_, rcond=None)

    if rk == 3:
        tfm = np.float32([
            [A[0, 0], A[1, 0], A[2, 0]],
            [A[0, 1], A[1, 1], A[2, 1]]
        ])
    elif rk == 2:
        tfm = np.float32([
            [A[0, 0], A[1, 0], 0],
            [A[0, 1], A[1, 1], 0]
        ])
    return tfm


def warp_and_crop_face(src_img,
                       facial_pts,
                       reference_pts=None,
                       crop_size=(96, 112),
                       align_type='similarity'):
    """把原图按 5 点对齐并裁剪成标准人脸。

    Parameters
    ----------
    src_img : HxWx3 ndarray (BGR)
    facial_pts : (5,2) 或 (2,5)
        关键点，顺序必须与参考模板一致（图像左眼、右眼、鼻尖、左嘴角、右嘴角）
    reference_pts : (5,2) 或 None
        None 时按 crop_size 自动推导（正方形输出用 default_square=True）
    crop_size : (w, h)
    align_type : 'similarity' | 'cv2_affine' | 'affine'
    """
    if reference_pts is None:
        if crop_size[0] == 96 and crop_size[1] == 112:
            reference_pts = REFERENCE_FACIAL_POINTS
        elif crop_size[0] == crop_size[1]:
            # ← 本实验新增：正方形输出（如 112x112）直接取 default_square 模板
            reference_pts = get_reference_facial_points(crop_size, 0, (0, 0),
                                                        default_square=True)
        else:
            reference_pts = get_reference_facial_points(crop_size, 0, (0, 0), False)

    ref_pts = np.float32(reference_pts)
    ref_shp = ref_pts.shape
    if max(ref_shp) < 3 or min(ref_shp) != 2:
        raise FaceWarpException('reference_pts.shape must be (K,2) or (2,K) and K>2')
    if ref_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_shp = src_pts.shape
    if max(src_shp) < 3 or min(src_shp) != 2:
        raise FaceWarpException('facial_pts.shape must be (K,2) or (2,K) and K>2')
    if src_shp[0] == 2:
        src_pts = src_pts.T

    if src_pts.shape != ref_pts.shape:
        raise FaceWarpException('facial_pts and reference_pts must have the same shape')

    if align_type == 'cv2_affine':
        tfm = cv2.getAffineTransform(src_pts[0:3], ref_pts[0:3])
    elif align_type == 'affine':
        tfm = get_affine_transform_matrix(src_pts, ref_pts)
    elif align_type in ('similarity', 'smilarity'):   # 兼容原版拼写
        tfm = get_similarity_transform_for_cv2(src_pts, ref_pts)
    else:
        raise FaceWarpException(f'unknown align_type: {align_type}')

    return cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]),
                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

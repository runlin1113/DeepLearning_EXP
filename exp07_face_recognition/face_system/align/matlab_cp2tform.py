"""
[复用的第三方源码] face.evoLVe  ——  applications/align/matlab_cp2tform.py
来源：https://github.com/ZhaoJ9014/face.evoLVe （MIT License）
作用：实现 MATLAB `cp2tform(..., 'similarity')` 的等价算法，
      求解两点集之间的**相似变换**（旋转 + 均匀缩放 + 平移，可选镜像反射）。

保留原文件全部函数名与算法，仅做两处工程修复：
  1) `__str__` 里 `super.__str__` 写法非法 → 改为标准异常实现；
  2) 显式标注 np.linalg.lstsq 的 rcond 参数（NumPy>=2.0 默认行为调整，避免告警）。
"""
import numpy as np
from numpy.linalg import inv, norm, lstsq
from numpy.linalg import matrix_rank as rank


class MatlabCp2tormException(Exception):
    def __str__(self):
        return "In File {}:{}".format(__file__, super().__str__())


def tformfwd(trans, uv):
    """应用仿射变换 trans 到 uv（Kx2）。"""
    uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy = np.dot(uv, trans)
    return xy[:, 0:-1]


def tforminv(trans, uv):
    """应用 trans 的逆变换到 uv（Kx2）。"""
    Tinv = inv(trans)
    return tformfwd(Tinv, uv)


def findNonreflectiveSimilarity(uv, xy, options=None):
    """求解**非反射**相似变换（4 自由度：scale, rotation, tx, ty）。"""
    options = {'K': 2}
    K = options['K']
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))
    y = xy[:, 1].reshape((-1, 1))

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))
    v = uv[:, 1].reshape((-1, 1))
    U = np.vstack((u, v))

    if rank(X) >= 2 * K:
        r, _, _, _ = lstsq(X, U, rcond=None)
        r = np.squeeze(r)
    else:
        raise Exception("cp2tform: two Unique Points Req")

    sc, ss, tx, ty = r[0], r[1], r[2], r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss, sc, 0],
        [tx, ty, 1]
    ])
    T = inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])
    return T, Tinv


def findSimilarity(uv, xy, options=None):
    """求解相似变换：同时尝试"非反射"与"先镜像再求解"，取残差更小者。"""
    options = {'K': 2}
    trans1, trans1_inv = findNonreflectiveSimilarity(uv, xy, options)

    # 手动把目标点沿 Y 轴镜像，再求一次
    xyR = xy.copy()
    xyR[:, 0] = -1 * xyR[:, 0]
    trans2r, trans2r_inv = findNonreflectiveSimilarity(uv, xyR, options)

    TreflectY = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    trans2 = np.dot(trans2r, TreflectY)

    xy1 = tformfwd(trans1, uv)
    norm1 = norm(xy1 - xy)
    xy2 = tformfwd(trans2, uv)
    norm2 = norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = inv(trans2)
        return trans2, trans2_inv


def get_similarity_transform(src_pts, dst_pts, reflective=True):
    """返回 3x3 变换矩阵 (uv -> xy) 及其逆。"""
    if reflective:
        trans, trans_inv = findSimilarity(src_pts, dst_pts)
    else:
        trans, trans_inv = findNonreflectiveSimilarity(src_pts, dst_pts)
    return trans, trans_inv


def cvt_tform_mat_for_cv2(trans):
    """3x3 (行向量约定) → 2x3 (cv2.warpAffine 约定)。"""
    return trans[:, 0:2].T


def get_similarity_transform_for_cv2(src_pts, dst_pts, reflective=True):
    """
    求可直接喂给 cv2.warpAffine 的 2x3 矩阵：
        [x, y].T = cv2_trans * [u, v, 1].T
    """
    trans, trans_inv = get_similarity_transform(src_pts, dst_pts, reflective)
    return cvt_tform_mat_for_cv2(trans)


if __name__ == '__main__':
    # 原文件自带的自检样例（与 MATLAB cp2tform 输出对齐）
    u = [0, 6, -2]
    v = [0, 3, 5]
    x = [-1, 0, 4]
    y = [-1, -10, 4]
    uv = np.array((u, v)).T
    xy = np.array((x, y)).T

    print("\n--->uv:\n", uv)
    print("\n--->xy:\n", xy)
    trans, trans_inv = get_similarity_transform(uv, xy)
    print("\n--->trans matrix:\n", trans)
    print("\n--->trans_inv matrix:\n", trans_inv)
    print("\n---> tformfwd(trans, uv):\n", tformfwd(trans, uv))
    print("\n---> tforminv(trans, xy):\n", tforminv(trans, xy))

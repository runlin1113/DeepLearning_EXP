"""
底库（Gallery）：人脸注册、特征库持久化、1:1 验证、1:N 检索。

特征空间约定：所有特征均 L2 归一化，余弦相似度 = 内积。
    1:1 (verification) : sim(a, b) > τ  → 同一人
    1:N (identification): argmax_i sim(a, g_i)，且 max > τ → 命中，否则判"未知"
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import CFG, ensure_dir
from .utils import cosine_sim, imread, l2norm, list_images


class Gallery:
    def __init__(self, threshold: float | None = None):
        self.threshold = CFG.rec.match_threshold if threshold is None else float(threshold)
        self.names: List[str] = []                       # 身份名列表（索引即 id）
        self.id_feats: np.ndarray | None = None          # (M, D) 每个身份的"均值特征"
        self.face_feats: List[np.ndarray] = []           # 每个身份的全部单脸特征（列表）
        self.meta: Dict[str, dict] = {}

    # ------------------------------------------------------------- 注册
    def add_identity(self, name: str, feats: Sequence[np.ndarray], meta: dict | None = None):
        """用同一身份的若干张人脸特征注册。feats: [(D,), ...] 已归一化。"""
        if len(feats) == 0:
            raise ValueError(f"{name}: 没有可用特征")
        F = l2norm(np.stack([np.asarray(f, np.float32).ravel() for f in feats]))
        if name in self.names:
            i = self.names.index(name)
            F = np.vstack([self.face_feats[i], F])
            self.face_feats[i] = F
            self.id_feats[i] = l2norm(F.mean(0))
        else:
            self.names.append(name)
            self.face_feats.append(F)
            mean = l2norm(F.mean(0))[None, :]
            self.id_feats = mean if self.id_feats is None else np.vstack([self.id_feats, mean])
        if meta:
            self.meta.setdefault(name, {}).update(meta)

    def register_from_aligned_dir(self, root, rec, aligner=None, limit: int = 0):
        """从"已对齐"的目录注册：root/<name>/*.jpg（112x112）"""
        root = Path(root)
        for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
            imgs = list_images(sub)[:limit or None]
            crops = [imread(p) for p in imgs]
            crops = [c for c in crops if c is not None]
            if not crops:
                continue
            feats = rec.get_feat(crops)
            self.add_identity(sub.name, feats, {"n_images": len(crops), "source": str(sub)})
        return self

    def register_from_raw_dir(self, root, rec, aligner, limit: int = 0):
        """从"原始"目录注册：root/<name>/*.jpg，内部自动 检测→对齐→提特征。"""
        root = Path(root)
        for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
            imgs = list_images(sub)[:limit or None]
            crops = []
            for p in imgs:
                img = imread(p)
                if img is None:
                    continue
                afs = aligner.run(img)
                if not afs:
                    continue
                crops.append(max(afs, key=lambda a: a.face.area).crop)
            if not crops:
                print(f"[gallery] 跳过 {sub.name}：未检出人脸")
                continue
            self.add_identity(sub.name, rec.get_feat(crops),
                              {"n_images": len(crops), "source": str(sub)})
        return self

    # ------------------------------------------------------------- 查询
    def finalize(self):
        """一致性检查：保证 id_feats 与 names 对得上。"""
        assert self.id_feats is not None and len(self.id_feats) == len(self.names), \
            "底库为空或长度不一致"
        return self

    @property
    def size(self) -> int:
        return len(self.names)

    def search(self, feat: np.ndarray, topk: int = 5) -> List[Tuple[str, float]]:
        """1:N 检索，返回 [(name, sim), ...] 降序。"""
        if self.id_feats is None or len(self.names) == 0:
            return []
        sims = cosine_sim(feat, self.id_feats)[0]
        order = np.argsort(-sims)[:topk]
        return [(self.names[i], float(sims[i])) for i in order]

    def search_face_level(self, feat: np.ndarray, topk: int = 5) -> List[Tuple[str, float]]:
        """逐脸检索后按身份取 top-k 均分，对多姿态注册更鲁棒（可选策略）。"""
        if not self.names:
            return []
        scores = []
        for i, name in enumerate(self.names):
            s = cosine_sim(feat, self.face_feats[i])[0]
            k = min(3, len(s))
            scores.append((name, float(np.sort(s)[-k:].mean())))
        scores.sort(key=lambda x: -x[1])
        return scores[:topk]

    def identify(self, feat: np.ndarray, threshold: float | None = None,
                 topk: int = 5, face_level: bool = False):
        """返回 (name 或 '未知', sim, topk列表)"""
        thr = self.threshold if threshold is None else float(threshold)
        res = self.search_face_level(feat, topk) if face_level else self.search(feat, topk)
        if not res:
            return "未知", 0.0, []
        name, sim = res[0]
        return (name if sim >= thr else "未知"), sim, res

    def verify(self, feat_a: np.ndarray, feat_b: np.ndarray, threshold: float | None = None):
        """1:1 比对，返回 (是否同一人, 余弦相似度)"""
        thr = self.threshold if threshold is None else float(threshold)
        sim = float(cosine_sim(feat_a, feat_b)[0, 0])
        return sim >= thr, sim

    def verify_images(self, path_a, path_b, rec, aligner, threshold=None):
        """两张原图直接做 1:1（内部自动检测+对齐）。"""
        fa = self._feat_of_image(path_a, rec, aligner)
        fb = self._feat_of_image(path_b, rec, aligner)
        if fa is None or fb is None:
            raise RuntimeError("图中未检出人脸")
        ok, sim = self.verify(fa, fb, threshold)
        return ok, sim

    @staticmethod
    def _feat_of_image(path, rec, aligner):
        img = imread(path)
        if img is None:
            return None
        afs = aligner.run(img)
        if not afs:
            return None
        return rec.get_feat([max(afs, key=lambda a: a.face.area).crop])[0]

    # ------------------------------------------------------------- 持久化
    def save(self, path=None):
        path = Path(path) if path else ensure_dir(CFG.data_dir / "gallery") / "gallery.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            names=np.array(self.names, dtype=object),
            id_feats=self.id_feats if self.id_feats is not None else np.zeros((0, 512), np.float32),
            face_feats=np.array([f for f in self.face_feats], dtype=object),
            meta=np.array(json.dumps(self.meta, ensure_ascii=False)),
            threshold=np.array([self.threshold], np.float32),
        )
        return path

    @classmethod
    def load(cls, path):
        d = np.load(Path(path), allow_pickle=True)
        g = cls(threshold=float(d["threshold"][0]) if "threshold" in d else None)
        g.names = [str(n) for n in d["names"]]
        g.id_feats = d["id_feats"].astype(np.float32)
        g.face_feats = [np.asarray(f, np.float32) for f in d["face_feats"]]
        g.meta = json.loads(str(d["meta"])) if "meta" in d else {}
        return g

    def summary(self) -> str:
        lines = [f"底库身份数: {self.size}  阈值 τ={self.threshold:.2f}"]
        for n in self.names:
            m = self.meta.get(n, {})
            lines.append(f"  - {n}: {len(self.face_feats[self.names.index(n)])} 张特征"
                         f" (来源 {m.get('source', '-')})")
        return "\n".join(lines)


def evaluate_1n(gallery: Gallery, probes: List[Tuple[str, np.ndarray]],
                threshold: float | None = None, topk: int = 3):
    """1:N 评测。probes = [(真实身份, 特征), ...]，返回 (rank1, rank_topk, 误接受率, 明细)"""
    rank1 = 0
    rankk = 0
    fa = 0
    details = []
    unknown_cnt = 0
    for truth, feat in probes:
        name, sim, res = gallery.identify(feat, threshold=threshold, topk=topk)
        ranking = [r[0] for r in res]
        hit1 = (name == truth)
        hitk = truth in ranking
        rank1 += int(hit1)
        rankk += int(hitk)
        if truth == "未知":
            unknown_cnt += 1
            fa += int(name != "未知")     # 陌生人被误认
        details.append({"truth": truth, "pred": name, "sim": sim,
                        "ranking": [(n, round(s, 4)) for n, s in res]})
    n = max(1, len(probes))
    return {
        "n_probes": len(probes),
        "rank1": rank1 / n,
        f"rank{topk}": rankk / n,
        "far_on_unknown": (fa / unknown_cnt) if unknown_cnt else None,
        "details": details,
    }

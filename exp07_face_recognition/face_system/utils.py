"""
通用工具：中文路径读写、中文文字绘制、计时、余弦相似度、可视化拼图。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

# ----------------------------------------------------------------- 中文路径 IO
def imread(path) -> np.ndarray | None:
    """cv2.imread 在 Windows 下读不了含中文的路径，用 imdecode 绕过。"""
    path = str(path)
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def imwrite(path, img) -> bool:
    """同样绕过中文路径写文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def list_images(root, recursive: bool = False) -> list[Path]:
    from .config import IMG_EXTS
    root = Path(root)
    it = root.rglob("*") if recursive else root.glob("*")
    return sorted([p for p in it if p.is_file() and p.suffix.lower() in IMG_EXTS])


# ------------------------------------------------------------------ 中文绘制
_FONT_CACHE = {}
_FONT_CANDIDATES = [
    r"C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
    r"C:/Windows/Fonts/simhei.ttf",    # 黑体
    r"C:/Windows/Fonts/simsun.ttc",
    "/System/Library/Fonts/PingFang.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


def _get_font(size: int):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    from PIL import ImageFont
    for p in _FONT_CANDIDATES:
        if Path(p).exists():
            f = ImageFont.truetype(p, size)
            _FONT_CACHE[size] = f
            return f
    f = ImageFont.load_default()
    _FONT_CACHE[size] = f
    return f


_TEXT_PATCH_CACHE = {}


def _text_patch(text: str, size: int, color, alpha: int = 150):
    """把一段文字渲染成小 RGBA 贴片并缓存（视频里每帧都写字，PIL 整图转换太慢）。"""
    key = (text, size, tuple(int(c) for c in color), alpha)
    hit = _TEXT_PATCH_CACHE.get(key)
    if hit is not None:
        return hit
    from PIL import Image, ImageDraw
    font = _get_font(size)
    dummy = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    l, t, r, b = dummy.textbbox((0, 0), text, font=font)
    w, h = max(1, r - l + 8), max(1, b - t + 8)
    img = Image.new("RGBA", (w, h), (0, 0, 0, alpha))
    d = ImageDraw.Draw(img)
    d.text((4 - l, 4 - t), text, font=font,
           fill=(int(color[2]), int(color[1]), int(color[0]), 255))
    arr = np.array(img)
    bgr = arr[:, :, :3].astype(np.float32)
    a = (arr[:, :, 3:4].astype(np.float32) / 255.0)
    _TEXT_PATCH_CACHE[key] = (bgr, a, w, h)
    return _TEXT_PATCH_CACHE[key]


def fit_font_size(text: str, max_w: int, size: int = 22, min_size: int = 8) -> int:
    """把字号逐步缩小到文字宽度不超过 max_w。

    用于"上图下文"的窄贴片：身份名长短不一（`Ariel_Sharon` vs
    `Arnold_Schwarzenegger`），固定字号必然裁字，这里按实测像素宽自适应。
    """
    s = int(size)
    while s > min_size:
        if _text_patch(str(text), s, (255, 255, 255))[2] <= max_w:
            break
        s -= 1
    return s


def put_text_scale(img, text, org, color=(0, 255, 0), size=22):
    """带半透明底的文字（中文安全、带缓存，适合视频逐帧调用）。org 为左上角。"""
    if not text:
        return img
    bgr, a, w, h = _text_patch(str(text), int(size), color)
    x, y = int(org[0]), int(org[1])
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return img
    sx, sy = x0 - x, y0 - y
    patch = bgr[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    pa = a[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - pa) + patch * pa).astype(np.uint8)
    return img


def draw_text_cn(img: np.ndarray, text: str, org, color=(0, 255, 0),
                 size: int = 22, bg: bool = True, bg_color=(0, 0, 0)) -> np.ndarray:
    """在 BGR 图上写中文（cv2.putText 不支持中文，走 PIL + 缓存贴片）。"""
    if not text:
        return img
    bgr, a, w, h = _text_patch(str(text), int(size), color, 160 if bg else 0)
    if not bg:
        a = np.where(a > 0, 1.0, 0.0).astype(np.float32)
    x, y = int(org[0]), int(org[1])
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return img
    sx, sy = x0 - x, y0 - y
    patch = bgr[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    pa = a[sy:sy + (y1 - y0), sx:sx + (x1 - x0)]
    roi = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = (roi * (1 - pa) + patch * pa).astype(np.uint8)
    return img


# ------------------------------------------------------------------- 数值工具
def l2norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a:(D,)或(N,D), b:(M,D) -> (N,M) 余弦相似度矩阵"""
    a = np.atleast_2d(l2norm(np.asarray(a, np.float32)))
    b = np.atleast_2d(l2norm(np.asarray(b, np.float32)))
    return a @ b.T


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """xywh 格式 IoU"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return float(inter / (aw * ah + bw * bh - inter + 1e-9))


def center_dist(a: Sequence[float], b: Sequence[float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return float(np.hypot(ax + aw / 2 - (bx + bw / 2), ay + ah / 2 - (by + bh / 2)))


class Timer:
    """上下文计时器：with Timer() as t: ... ; t.ms"""

    def __init__(self):
        self.ms = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.ms = (time.perf_counter() - self._t0) * 1000


class FPSMeter:
    def __init__(self, window: int = 30):
        self.window = window
        self.ts: list[float] = []

    def tick(self) -> float:
        now = time.perf_counter()
        self.ts.append(now)
        if len(self.ts) > self.window:
            self.ts.pop(0)
        if len(self.ts) < 2:
            return 0.0
        dt = self.ts[-1] - self.ts[0]
        return (len(self.ts) - 1) / dt if dt > 0 else 0.0


# ------------------------------------------------------------------- 可视化
def make_grid(images: Iterable[np.ndarray], cols: int = 5, pad: int = 6,
              bg=(30, 30, 30), titles: Sequence[str] | None = None,
              title_size: int = 16) -> np.ndarray:
    """把若干张（可不等大）图片拼成网格，自动统一尺寸。"""
    imgs = [im for im in images if im is not None]
    if not imgs:
        return np.zeros((100, 100, 3), np.uint8)
    h = max(im.shape[0] for im in imgs)
    w = max(im.shape[1] for im in imgs)
    rows = (len(imgs) + cols - 1) // cols
    canvas = np.full((rows * (h + pad + 24) + pad, cols * (w + pad) + pad, 3), bg, np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad + 24)
        x = pad + c * (w + pad)
        canvas[y:y + im.shape[0], x:x + im.shape[1]] = im
        if titles and i < len(titles):
            canvas = draw_text_cn(canvas, str(titles[i]), (x + 2, y + im.shape[0] + 2),
                                  color=(200, 200, 200), size=title_size, bg=False)
    return canvas


def draw_face_box(img, bbox, landmarks=None, color=(0, 255, 0), label=None,
                  thickness: int | None = None, label_size: int = 20):
    """画人脸框 + 5 关键点 + 标签（中文安全）。"""
    x, y, w, h = [int(v) for v in bbox]
    if thickness is None:
        thickness = max(1, int(round(min(img.shape[:2]) / 300)))
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
    if landmarks is not None:
        for j, (px, py) in enumerate(np.asarray(landmarks).reshape(-1, 2)):
            cv2.circle(img, (int(px), int(py)), max(2, thickness + 1),
                       (0, 255, 255) if j < 2 else (255, 128, 0), -1)
    if label:
        ty = y - label_size - 8
        if ty < 0:
            ty = y + 2
        img = put_text_scale(img, label, (x + 2, ty), color=color, size=label_size)
    return img


def setup_mpl_cn(font_size: int = 10):
    """让 matplotlib 正常显示中文（默认 DejaVu Sans 没有 CJK 字形，会画成方框）。"""
    try:
        import matplotlib
        matplotlib.rcParams["font.sans-serif"] = [
            "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
            "PingFang SC", "DejaVu Sans",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False
        matplotlib.rcParams["font.size"] = font_size
    except Exception:
        pass

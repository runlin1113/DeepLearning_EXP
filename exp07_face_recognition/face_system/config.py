"""
全局配置：路径、模型、阈值集中管理。
所有脚本通过 `from face_system.config import CFG` 读取，避免硬编码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------- 目录布局
ROOT = Path(__file__).resolve().parent.parent          # 第七节实验/
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

RAW_FACE_DIR = DATA_DIR / "images" / "self"      # 本人 15 张原始照片
LFW_DIR = DATA_DIR / "lfw20"                           # 公开小样本人脸库（训练/评估）
LFW_ALIGNED_DIR = DATA_DIR / "lfw20_aligned"           # 经本工程对齐后的 112x112
GALLERY_DIR = DATA_DIR / "gallery"                     # 注册底库（对齐后的人脸）

for _d in (MODEL_DIR, OUTPUT_DIR, DATA_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class DetectorCfg:
    """人脸检测器配置"""
    name: str = "yunet"                                # yunet | centerface
    yunet_model: str = str(MODEL_DIR / "face_detection_yunet_2023mar.onnx")
    centerface_model: str = str(MODEL_DIR / "centerface.onnx")
    input_size: tuple = (640, 640)                     # (w, h) 推理输入尺寸
    score_threshold: float = 0.6
    nms_threshold: float = 0.3
    top_k: int = 5000
    max_side: int = 1280                               # 大图先降采样再检测（加速）
    min_face: int = 24                                 # 过滤过小人脸（像素）


@dataclass
class AlignCfg:
    """对齐配置（复用 face.evoLVe 的相似变换 + 参考点）"""
    crop_size: int = 112                               # ArcFace 标准输入 112x112
    align_type: str = "similarity"                     # similarity | cv2_affine | affine
    auto_fix_reflection: bool = True                   # 自动纠正左右眼顺序导致的镜像


@dataclass
class RecCfg:
    """特征提取 / 比对配置"""
    model: str = str(MODEL_DIR / "w600k_r50.onnx")     # ArcFace(ResNet50, Glint360K)
    feat_dim: int = 512
    # 余弦阈值：同人判定。w600k_r50 在 LFW 类数据上 1e-3 FAR 约 0.35~0.40
    match_threshold: float = 0.35
    # 视频多帧融合
    ema_alpha: float = 0.6                             # 特征 EMA 融合系数（越大越信当前帧）
    track_buffer: int = 30                             # 轨迹丢失后保留帧数
    min_track_hits: int = 3                            # 连续命中多少帧才输出姓名
    vote_window: int = 15                              # 时序投票窗口


@dataclass
class TrainCfg:
    """训练实验配置"""
    dataset: str = str(LFW_ALIGNED_DIR)
    image_size: int = 112
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 5e-4
    scale: float = 30.0                                # 超球半径 s
    margin: float = 0.35                               # 角度/余弦 margin m
    depth: int = 18                                    # 小样本从零训：iResNet-18
    num_workers: int = 0                               # Windows 建议 0
    seed: int = 42


@dataclass
class CFG:
    detector: DetectorCfg = field(default_factory=DetectorCfg)
    align: AlignCfg = field(default_factory=AlignCfg)
    rec: RecCfg = field(default_factory=RecCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    root: Path = ROOT
    model_dir: Path = MODEL_DIR
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    raw_face_dir: Path = RAW_FACE_DIR
    lfw_dir: Path = LFW_DIR
    lfw_aligned_dir: Path = LFW_ALIGNED_DIR
    gallery_dir: Path = GALLERY_DIR


CFG = CFG()

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_cuda_env():
    """把 CUDA 运行库 DLL 目录塞进 PATH（Windows + pip 版 torch 常见问题）。"""
    if os.name != "nt":
        return
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia")
        if spec and spec.submodule_search_locations:
            base = Path(list(spec.submodule_search_locations)[0])
            for sub in base.glob("*/bin"):
                os.add_dll_directory(str(sub))
    except Exception:
        pass


def resolve_onnx_providers(prefer_gpu: bool = False):
    """选择 onnxruntime 执行后端，GPU 不可用自动回落 CPU。"""
    import onnxruntime as ort
    avail = ort.get_available_providers()
    if prefer_gpu and "CUDAExecutionProvider" in avail:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]

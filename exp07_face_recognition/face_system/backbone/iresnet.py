"""
PyTorch 版 iResNet 骨干（训练实验用，结构对齐 insightface 的 iresnet）。

- 用于 step7 的"Softmax / CosFace / ArcFace"对照训练（小数据从零训，depth=18 足够）
- 也可加载 insightface 官方 .pth（Glint360K-R100）权重做纯 PyTorch 推理

尺寸关系：112x112 输入 → 3x3/stride1 茎层 → stage1 3x3/s1 → stage2 s2 → stage3 s2 →
          stage4 s2 → 7x7 / 1x1 → FC(512)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BNConv(nn.Module):
    """Conv2d(no bias) + BN + PReLU，insightface 的标准单元。"""

    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_c),
            nn.PReLU(out_c),
        )

    def forward(self, x):
        return self.body(x)


class bottleneck_IR(nn.Module):
    def __init__(self, in_c, depth, stride):
        super().__init__()
        if in_c == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride=stride) if stride != 1 else nn.Identity()
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_c, depth, 1, stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            BNConv(in_c, depth, (3, 3), (1, 1), 1),
            BNConv(depth, depth, (3, 3), stride, 1),
        )

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class bottleneck_IR_SE(nn.Module):
    """IR-SE：带 Squeeze-Excitation，R50/R100 官方权重用的是这个版本。"""

    def __init__(self, in_c, depth, stride):
        super().__init__()
        if in_c == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride=stride) if stride != 1 else nn.Identity()
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_c, depth, 1, stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            BNConv(in_c, depth, (3, 3), (1, 1), 1),
            BNConv(depth, depth, (3, 3), stride, 1),
            SEModule(depth, 16),
        )

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        m = self.avg_pool(x)
        m = self.fc2(self.relu(self.fc1(m)))
        return x * self.sigmoid(m)


class IResNet(nn.Module):
    """返回 512 维**未归一化**特征的骨干。"""

    def __init__(self, units=(2, 2, 2, 2), block=bottleneck_IR,
                 dropout=0.0, emb_dim=512, input_size=112):
        super().__init__()
        self.input_size = input_size
        self.emb_dim = emb_dim
        self.block = block
        self.units = units

        # 茎层：第一个 3x3 stride1 提特征，第二个 3x3 stride2 直接降一半分辨率
        # （112 -> 56），于是 4 个 stage 之后特征图为 112/16 = 7，与 insightface 一致
        self.input_layer = nn.Sequential(
            BNConv(3, 64, (3, 3), (1, 1), 1),
            BNConv(64, 64, (3, 3), (2, 2), 1),
        )
        # 4 个 stage，通道 64/128/256/512，除第一段外每段首个残差块做 stride=2 下采样
        chans = [64, 64, 128, 256, 512]
        strides = [1, 2, 2, 2]
        seq = []
        for si, (n, s) in enumerate(zip(units, strides), start=1):
            cin = chans[si - 1]
            cout = chans[si]
            for i in range(n):
                seq.append(block(cin if i == 0 else cout, cout, s if i == 0 else 1))
        self.body = nn.Sequential(*seq)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(512 * (input_size // 16) * (input_size // 16), emb_dim),
            nn.BatchNorm1d(emb_dim, affine=False),
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        return self.output_layer(x)


def iresnet18(**kw):
    return IResNet((2, 2, 2, 2), bottleneck_IR, **kw)


def iresnet34(**kw):
    return IResNet((3, 4, 6, 3), bottleneck_IR, **kw)


def iresnet50(**kw):
    return IResNet((3, 4, 14, 3), bottleneck_IR_SE, **kw)


def iresnet100(**kw):
    return IResNet((3, 13, 30, 3), bottleneck_IR_SE, **kw)


_DEPTH_MAP = {18: iresnet18, 34: iresnet34, 50: iresnet50, 100: iresnet100}


def build_backbone(depth: int = 18, **kw) -> IResNet:
    if depth not in _DEPTH_MAP:
        raise ValueError(f"depth 必须是 {list(_DEPTH_MAP)} 之一")
    return _DEPTH_MAP[depth](**kw)


def load_insightface_state(model: IResNet, pth_path: str, strict: bool = False):
    """加载 insightface 官方 .pth（只取骨干部分，丢弃最后的分类头）。"""
    sd = torch.load(pth_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    own = model.state_dict()
    filtered = {k: v for k, v in sd.items()
                if k in own and own[k].shape == v.shape}
    missing = [k for k in own if k not in filtered]
    model.load_state_dict(filtered, strict=False)
    if not strict and missing:
        print(f"[iResNet] 已加载 {len(filtered)} 个张量，未匹配 {len(missing)} 个"
              f"（多为 FC 分类头，属正常）")
    return model

# -*- coding: utf-8 -*-
"""
把 dataset/photo_test/*.json (X-AnyLabeling 标注) 转为 YOLO 标签，
并组装成独立测试集 dataset/test/ （images + labels + data_test.yaml）。
纯数据转换，不依赖 ultralytics，安全。
"""

# 统一工作目录：以下相对路径均以本文件所在目录为基准，克隆到任意位置可直接运行
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, json, glob, shutil

SRC = "dataset/photo_test"
DST = "dataset/test"
IMG_DST = os.path.join(DST, "images")
LBL_DST = os.path.join(DST, "labels")
os.makedirs(IMG_DST, exist_ok=True)
os.makedirs(LBL_DST, exist_ok=True)

jsons = sorted(glob.glob(os.path.join(SRC, "*.json")))
print(f"找到 {len(jsons)} 个标注文件")
total_boxes = 0
for jp in jsons:
    base = os.path.splitext(os.path.basename(jp))[0]
    d = json.load(open(jp, encoding="utf-8"))
    W, H = d["imageWidth"], d["imageHeight"]
    lines = []
    for sh in d.get("shapes", []):
        if sh.get("shape_type", "rectangle") not in ("rectangle",):
            continue
        pts = sh.get("points", [])
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x1, x2 = max(0, min(xs)), min(W, max(xs))
        y1, y2 = max(0, min(ys)), min(H, max(ys))
        bw, bh = x2 - x1, y2 - y1
        if bw <= 1 or bh <= 1:
            continue
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw / W:.6f} {bh / H:.6f}")
    with open(os.path.join(LBL_DST, base + ".txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    # 复制图片
    ip = os.path.join(SRC, base + ".jpg")
    if os.path.exists(ip):
        shutil.copy(ip, os.path.join(IMG_DST, base + ".jpg"))
    total_boxes += len(lines)
    print(f"  {base}: {len(lines)} 框")

with open(os.path.join(DST, "data_test.yaml"), "w", encoding="utf-8") as f:
    f.write(f"path: {os.path.abspath(DST)}\n")
    f.write("train: images\n")
    f.write("val: images\n")
    f.write("nc: 1\nnames: ['start']\n")

print(f"\n完成：{len(jsons)} 张测试图，共 {total_boxes} 个框")
print(f"输出：{os.path.abspath(DST)} (images/ labels/ data_test.yaml)")

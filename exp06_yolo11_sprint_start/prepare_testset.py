# -*- coding: utf-8 -*-
"""
把 photo_backup_20261019 中今天(2026-10-20)新增的 10 张图，
转为规范命名 test_01.jpg ~ test_10.jpg，复制到一个独立的测试集文件夹
dataset/photo_test/（与训练数据完全隔离），供 X-AnyLabeling 标注使用。

原图保留在备份目录，不做删除。
"""

# 统一工作目录：以下相对路径均以本文件所在目录为基准，克隆到任意位置可直接运行
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, glob, shutil, datetime

ROOT = "photo_backup_20261019"
DST = "dataset/photo_test"
os.makedirs(DST, exist_ok=True)

# 选取 2026-10-20 12:00 之后修改的（即本次新增的 10 张）
threshold = datetime.datetime(2026, 10, 20, 12, 0).timestamp()
cands = [f for f in glob.glob(os.path.join(ROOT, "*.jpg")) if os.path.getmtime(f) >= threshold]
cands.sort()

mapping = []
for i, f in enumerate(cands, 1):
    new = f"test_{i:02d}.jpg"
    shutil.copy(f, os.path.join(DST, new))
    mapping.append(f"{os.path.basename(f)}  ->  {new}")

with open(os.path.join(DST, "_test_rename_map.txt"), "w", encoding="utf-8") as mf:
    mf.write("# 测试集重命名映射（原图位于 photo_backup_20261019，本目录仅作标注/评估用）\n")
    mf.write("\n".join(mapping) + "\n")

print(f"已复制 {len(cands)} 张到 {DST}")
for m in mapping:
    print(" ", m)

# -*- coding: utf-8 -*-
"""
为小数据集生成「抗泄漏」的数据划分：
1) 用感知哈希(aHash)把连拍近重复图聚成组
2) 按组做 K 折交叉验证(K-Fold) + 一个分组单划分(80/20)
3) 输出 fold/分组 的 txt 图集列表 + 对应 data_*.yaml
生成目录: dataset/kfold/
"""

# 统一工作目录：以下相对路径均以本文件所在目录为基准，克隆到任意位置可直接运行
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os, glob, json, shutil
from PIL import Image

ROOT = "dataset"
KFOLD_DIR = os.path.join(ROOT, "kfold")
N_FOLDS = 5
GROUP_THRESH = 6  # aHash 汉明距离 <=6 视为同一组的近重复图

# ---------- 1. 收集 52 张原图，建立 kfold 扁平 images/labels ----------
imgs = sorted(glob.glob(os.path.join(ROOT, "photo", "*.jpg")))
os.makedirs(os.path.join(KFOLD_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(KFOLD_DIR, "labels"), exist_ok=True)

def ahash(path, size=8):
    im = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    px = list(im.getdata())
    avg = sum(px) / len(px)
    bits = "".join("1" if p > avg else "0" for p in px)
    return int(bits, 2)

hashes = {}
for p in imgs:
    b = os.path.splitext(os.path.basename(p))[0]
    hashes[b] = ahash(p)
    shutil.copy(p, os.path.join(KFOLD_DIR, "images", b + ".jpg"))
    placed = False
    for sub in ("train", "val"):
        src = os.path.join(ROOT, "labels", sub, b + ".txt")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(KFOLD_DIR, "labels", b + ".txt"))
            placed = True
            break
    if not placed:
        open(os.path.join(KFOLD_DIR, "labels", b + ".txt"), "w").close()  # 兜底空标签

# ---------- 2. 并查集聚类近重复组 ----------
bases = list(hashes.keys())
parent = {b: b for b in bases}
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
for i in range(len(bases)):
    for j in range(i + 1, len(bases)):
        if (hashes[bases[i]] ^ hashes[bases[j]]).bit_count() <= GROUP_THRESH:
            union(bases[i], bases[j])
cluster = {}
for b in bases:
    cluster.setdefault(find(b), []).append(b)
groups = list(cluster.values())
print(f"总图 {len(bases)} 张 -> 聚成 {len(groups)} 组, 各组大小: {sorted(len(g) for g in groups)}")

# ---------- 3. 贪心均衡分组 K 折 ----------
fold_groups = [[] for _ in range(N_FOLDS)]
fold_sizes = [0] * N_FOLDS
for g in sorted(groups, key=len, reverse=True):
    tidx = min(range(N_FOLDS), key=lambda k: fold_sizes[k])
    fold_groups[tidx].append(g)
    fold_sizes[tidx] += len(g)

def write_list(path, items):
    with open(path, "w") as f:
        f.write("\n".join(os.path.abspath(os.path.join(ROOT, "kfold", "images", b + ".jpg")) for b in items) + "\n")

for k in range(N_FOLDS):
    val = [b for g in fold_groups[k] for b in g]
    train = [b for kk in range(N_FOLDS) if kk != k for g in fold_groups[kk] for b in g]
    write_list(os.path.join(KFOLD_DIR, f"fold_{k}_train.txt"), train)
    write_list(os.path.join(KFOLD_DIR, f"fold_{k}_val.txt"), val)
    with open(os.path.join(KFOLD_DIR, f"data_fold_{k}.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(ROOT)}\n")
        f.write(f"train: kfold/fold_{k}_train.txt\n")
        f.write(f"val: kfold/fold_{k}_val.txt\n")
        f.write("nc: 1\nnames: ['start']\n")
    print(f"Fold {k}: train {len(train)}  val {len(val)}")

# ---------- 4. 分组单划分 80/20 (非 kfold 的易用升级版) ----------
val_single, train_single = [], []
target = int(0.2 * len(bases))
for g in sorted(groups, key=len, reverse=True):
    if sum(len(x) for x in val_single) + len(g) <= target:
        val_single.append(g)
    else:
        train_single.append(g)
val_flat = [b for g in val_single for b in g]
train_flat = [b for g in train_single for b in g]
write_list(os.path.join(KFOLD_DIR, "grouped_train.txt"), train_flat)
write_list(os.path.join(KFOLD_DIR, "grouped_val.txt"), val_flat)
with open(os.path.join(KFOLD_DIR, "data_grouped.yaml"), "w") as f:
    f.write(f"path: {os.path.abspath(ROOT)}\n")
    f.write("train: kfold/grouped_train.txt\n")
    f.write("val: kfold/grouped_val.txt\n")
    f.write("nc: 1\nnames: ['start']\n")
print(f"分组单划分: train {len(train_flat)}  val {len(val_flat)}")

# ---------- 5. 全部数据(用于最终部署模型, 不预留验证) ----------
write_list(os.path.join(KFOLD_DIR, "all_train.txt"),
           [b for b in bases])
with open(os.path.join(KFOLD_DIR, "data_all.yaml"), "w") as f:
    f.write(f"path: {os.path.abspath(ROOT)}\n")
    f.write("train: kfold/all_train.txt\n")
    f.write("val: kfold/all_train.txt\n")  # 监控用, 训练集本身
    f.write("nc: 1\nnames: ['start']\n")
print("全部数据划分: 52 张用于训练(最终部署模型)")

# ---------- 6. 保存分组信息供查阅 ----------
with open(os.path.join(KFOLD_DIR, "groups.json"), "w") as f:
    json.dump({
        "n_images": len(bases), "n_groups": len(groups),
        "group_sizes": sorted(len(g) for g in groups),
        "groups": groups,
        "folds": [[b for g in fold_groups[k] for b in g] for k in range(N_FOLDS)],
    }, f, indent=2, ensure_ascii=False)
print("\n已生成 dataset/kfold/ 下的分组 K 折 + 分组单划分 + 全量划分")

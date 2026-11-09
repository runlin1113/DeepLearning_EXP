# -*- coding: utf-8 -*-
"""分析各折验证集的图像/框特征，解释 K 折指标方差来源。"""

# 统一工作目录：以下相对路径均以本文件所在目录为基准，克隆到任意位置可直接运行
import os as _os
_os.chdir(_os.path.dirname(_os.path.abspath(__file__)))
import os

KF = "dataset/kfold"
for k in range(5):
    val = [l.strip() for l in open(f"{KF}/fold_{k}_val.txt", encoding="utf-8") if l.strip()]
    total, areas, small = 0, [], 0
    for p in val:
        lp = p.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        if os.path.exists(lp):
            for ln in open(lp, encoding="utf-8"):
                parts = ln.split()
                if len(parts) == 5:
                    w, h = float(parts[3]), float(parts[4])
                    areas.append(w * h)
                    total += 1
                    if w * h < 0.02:   # 小目标(<2%面积)
                        small += 1
    if areas:
        print(f"Fold {k}: val={len(val):2d}图 框={total:2d}  "
              f"平均框面积={sum(areas)/len(areas):.4f} 小目标占比={small/total:.0%}")
    else:
        print(f"Fold {k}: val={len(val)}图 0框(异常)")

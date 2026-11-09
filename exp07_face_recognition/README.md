# 实验七 · 人脸检测与动态人脸识别系统

一条完整的应用级链路：**图像/视频 → 人脸检测 → 相似变换对齐 → ArcFace 特征 → 余弦比对（1:1 / 1:N）→ 视频追踪与多帧融合**。全部在 CPU 上跑，16.5 FPS。

## 五段链路

| 阶段 | 模块 | 输入 → 输出 | 实现 |
|---|---|---|---|
| ① 检测 | `face_system/detectors/yunet.py` | `H×W×3` → `List[Face]`（框 + 5 关键点 + 置信度） | YuNet（ONNX，228 KB），cv2 与 onnxruntime 双后端 |
| ② 对齐 | `face_system/align/align_pipeline.py` | `图 + Face` → `112×112×3` | 复用 face.evoLVe 的相似变换，**改造后的核心** |
| ③ 特征 | `face_system/backbone/arcface_onnx.py` | `N×112×112` → `N×512`（已 L2 归一化） | ArcFace ResNet-50（`w600k_r50.onnx`，167 MB） |
| ④ 比对 | `face_system/gallery.py` | `512` 向量 → `(姓名, 相似度)` | 归一化后内积即余弦；`sim < τ` 判「未知」 |
| ⑤ 时序 | `face_system/tracker.py` | 本帧 `faces + feats` → `List[Track]` | IoU 关联 + EMA 特征融合 + 15 帧投票 |

## 目录

```
face_system/
├── config.py                 全部超参集中管理（阈值、crop 尺寸、EMA α…）
├── utils.py                  中文路径读写、余弦、IoU、中文字幕渲染
├── detectors/                base.py 定义 Face 数据结构；yunet.py 主检测器；centerface.py 备选
├── align/                    matlab_cp2tform.py（解相似变换）+ align_trans.py（模板与 warp）
│                             + align_pipeline.py（把检测器接进去，本实验的改造点）
├── backbone/                 arcface_onnx.py（推理）；iresnet.py（训练用骨干）
├── gallery.py                底库：注册、检索、1:1 验证、存取 npz
└── tracker.py                轨迹关联与身份投票
scripts/                      step0 下载权重 → step1 自检 → step2 检测对齐 → step3 建库
                              → step4 静态测试 → step5 视频 → step6 评估
train/                        三损失对照训练（softmax / cosface / arcface）
app/attendance_demo.py        应用示例：把五段串成「刷脸打卡」
tools/                        md→docx 报告生成、流程图绘制
代码模块详解.md                逐模块手册（职责 / 输入输出 / 关键代码 / 坑）
```

## 环境

```bash
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install numpy opencv-python onnxruntime pillow matplotlib tqdm
pip install torch torchvision                            # 仅 train/ 需要
python scripts/step0_download_models.py                  # 自动下载并校验两个 onnx
python scripts/step1_env_check.py                        # 端到端冒烟
```

`step0` 会从多个镜像源尝试下载 YuNet 与 ArcFace，并用 sha256 校验。若网络不通，手动放到 `models/` 下同名即可。

## 数据准备

仓库不含任何人脸数据。两步：

**1) 注册自己的身份**（1:1 与 1:N 都要用）

```
data/images/self/            8~15 张本人照片，jpg/png，正脸为主
```

目录名 `self` 就是底库里的身份名，想换名字改 `face_system/config.py` 的 `RAW_FACE_DIR` 一行即可（`scripts/step3` 里身份名同样取自该目录）。

**2) 评估用的公开人脸库**（可选，但没它跑不出 EER / Rank-1）

```
data/lfw20/train/<身份>/*.jpg     每身份 15 张
data/lfw20/valid/<身份>/*.jpg     每身份 5 张
```

LFW（Labeled Faces in the Wild）可从官网申请下载，按上面的两级目录摆放即可。实验里用了 62 个身份子集（930 训练 / 310 验证）。

## 运行

```bash
python scripts/step1_env_check.py        # 环境 + 端到端冒烟
python scripts/step2_detect_align.py     # 检测与对齐，出图 + 批量对齐 LFW
python scripts/step3_build_gallery.py    # 建底库（本人 8 注册 / 7 探测，划分写入 json）
python scripts/step4_static_test.py      # 静态 1:1、1:N、陌生人 FAR
python scripts/step5_video_demo.py --make-test-video --hard-video
python scripts/step5_video_demo.py --mode track_fusion     # 动态识别
python scripts/step6_evaluate.py         # 五组定量评估 + 全部图表
python train/train_margin_compare.py -e 40                 # 三损失对照（需 torch）
python app/attendance_demo.py --video outputs/step5/test_video.mp4   # 应用示例
```

视频三模式对比（同一段合成视频）：

```bash
python scripts/step5_video_demo.py --mode frame           # 逐帧独立判定（基线）
python scripts/step5_video_demo.py --mode track_nofusion  # 加追踪，不融合特征
python scripts/step5_video_demo.py --mode track_fusion    # 追踪 + EMA + 投票（推荐）
```

## 对齐为什么要改、改了什么

课程要求复用 face.evoLVe 的 `application/align`，但原代码是为 MTCNN 写的（96×112 模板、检测器自己给 5 点）。本工程做了三处改造：

**① 换检测器**：把 MTCNN 换成 YuNet，只保留「5 个关键点」这个接口约定。`detectors/base.py` 的 `Face(bbox, score, landmarks)` 是全链路唯一的数据契约，换检测器时下游一行不用改。

**② 换模板**：`REFERENCE_FACIAL_POINTS` 从 96×112 改为 **112×112**（ArcFace 的标准输入），参考点整体右移 8 px：

```python
ARCFACE_REF_112 = np.array([[38.29, 51.70], [73.53, 51.50], [56.03, 71.74],
                            [41.55, 92.37], [70.73, 92.20]], dtype=np.float32)
```

**③ 镜像自纠正**（原代码没有）：相似变换的 2×2 子块是 `s·R`，纯旋转 `det=+1`，含反射 `det=−1`。若检测器把左右眼顺序判反，解出的矩阵会带反射，**不报错但相似度悄悄下降**。所以解完检查行列式，为负就交换点位（`[1,0,2,4,3]`）重解一次：

```python
tfm = get_similarity_transform_for_cv2(pts5, REF112)
if np.linalg.det(tfm[:, :2]) < 0:                 # 含镜像 → 左右眼/嘴角判反
    tfm = get_similarity_transform_for_cv2(pts5[SWAP], REF112)
```

`align_trans.py` 另外修了两处：`crop_size` 为正方形时自动 `default_square=True`（原版会抛异常）、`align_type == 'cv2_affine'`（原版误用 `is` 比较字符串）。

## 结果

| 指标 | 数值 | 说明 |
|---|---|---|
| 检测率 | 100%（325 张） | 本人 15 + LFW 310；单张 8~10 ms |
| 对齐成功率 | 1240/1240 | 0 失败、0 次镜像纠正；双眼 y 差均值 0.30 px |
| 1:1 同人相似度 | 0.6673 ± 0.0924 | LFW 620 对 |
| 1:1 异人相似度 | 0.0059 ± 0.0589 | LFW 12000 对 |
| AUC / EER | **1.00000 / 0.029%** | 理论上限附近的分数 |
| 1:N Rank-1 | **100%** | 62 身份、310 张探测 |
| 陌生人 FAR | 0% | τ=0.35，90 张陌生脸全部拒识 |
| 对齐消融 | EER 0.031% vs 0.648% | 相似变换 vs 框直接拉伸，**差 20 倍** |
| 动态识别 | 稳定度 100%，0 次跳变，16.5 FPS | `track_fusion` 模式，250 帧合成视频 |
| 三损失对照 | 77.4% → 90.6% → 90.6% | Softmax → CosFace → ArcFace（Rank-1） |

最有信息量的一条：**对齐带来的增益大于换更大模型的增益**。同一张脸，用检测框直接 `resize` 到 112×112，EER 从 0.031% 恶化到 0.648%（20 倍），因为它允许横竖不等比缩放和剪切，把五官比例改了；相似变换只解 4 个自由度（旋转、等比缩放、平移），几何形状得以保留。

## 部署时的阈值怎么定

EER 最优阈值是 0.236，但实际部署取 **0.35**。原因：EER 假设同人/异人先验各半，而考勤场景里陌生人远多于本人，误接受（把路人认成队员）的代价远高于误拒绝（多看一眼）。抬高阈值等于用 TAR 换 FAR——实测 τ=0.35 时本人 TAR 仍为 100%，同时把 90 张陌生脸全挡在门外。

## 常见错误

| 现象 | 原因与处理 |
|---|---|
| `Can't read ONNX file`（OpenCV 5.x） | 新版 `ONNXImporter2` 读不了 YuNet 的旧算子；本工程 `backend='auto'` 会自动回退到 onnxruntime |
| `Expected shape {1,512} does not match {15,512}` | ArcFace onnx 的 batch 维写死为 1，必须逐块推理（`batch_size=8` 已处理） |
| `mat1 and mat2 shapes cannot be multiplied (32x100352 …)` | iResNet 茎层第二个 3×3 漏了 stride=2，特征图是 14×14 而非 7×7 |
| 视频只有 5~6 FPS | 每帧都对所有脸跑 ArcFace + 逐帧 PIL 整图渲染中文；改成「关键帧提特征 + 文字贴片缓存」后 16.5 FPS |
| 识别结果在「识别中 / 未知 / 名字」之间跳 | 投票窗口为 1，或 `min_track_hits` 太大；把 `vote_window` 调到 15、`min_track_hits=3` |
| 中文显示成方框 | matplotlib 需显式设置 CJK 字体（`setup_mpl_cn()`），OpenCV 不能直接画中文（用 PIL 贴片） |

## 引用

- 对齐代码改造自 [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe)（MIT），保留原文件头与函数名，改动处已在 `代码模块详解.md` 标注；
- 检测器 [YuNet](https://github.com/opencv/opencv_zoo)（OpenCV Zoo）；
- 特征提取 ArcFace `w600k_r50`（InsightFace Model Zoo），仅用于课程实验；
- LFW 数据集版权归原论文作者，仓库不再分发。

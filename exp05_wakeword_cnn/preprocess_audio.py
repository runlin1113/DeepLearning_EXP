import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局配置，一次设定，后续基本不用修改 =====================
# 路径基于本脚本所在目录，自动适配任意机器（不再写死学校电脑路径）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset")
SR = 16000
N_MELS = 40
FMAX = 8000
TARGET_TIME_STEPS = 32  # 固定时间帧，CNN输入维度固定
SAVE_ROOT = os.path.join(BASE_DIR, "preprocessed")
# ==============================================================================

def process_single_wav(wav_path: str):
    """加载单条wav，输出固定尺寸mel_db数组"""
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, fmax=FMAX)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 截断/补零到固定时间步
    curr_t = mel_db.shape[1]
    if curr_t > TARGET_TIME_STEPS:
        mel_db = mel_db[:, :TARGET_TIME_STEPS]
    else:
        pad_t = TARGET_TIME_STEPS - curr_t
        pad = np.zeros((N_MELS, pad_t))
        mel_db = np.concatenate([mel_db, pad], axis=1)
    return mel_db


def main():
    # 输出目录
    img_root = os.path.join(SAVE_ROOT, "mel_images")
    npy_root = os.path.join(SAVE_ROOT, "mel_npy")
    os.makedirs(img_root, exist_ok=True)
    os.makedirs(npy_root, exist_ok=True)

    # 自动读取所有类别（子文件夹名作为类别名称）
    class_names = [folder for folder in os.listdir(DATASET_ROOT)
                   if os.path.isdir(os.path.join(DATASET_ROOT, folder))]
    class_names.sort()
    print(f"✅ 检测到类别列表: {class_names}")
    print(f"类别总数: {len(class_names)}\n")

    # 保存类别映射文件，训练脚本直接读取，避免两处硬编码
    cls_map_path = os.path.join(SAVE_ROOT, "class_names.txt")
    with open(cls_map_path, "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print(f"类别名称已保存到 {cls_map_path}\n")

    total_count = 0
    for cls_name in class_names:
        cls_wav_dir = os.path.join(DATASET_ROOT, cls_name, "wav_out")
        if not os.path.exists(cls_wav_dir):
            print(f"⚠️ {cls_name} 下没有找到 wav_out 文件夹，跳过")
            continue

        # 为当前类别创建输出子文件夹
        cls_img_dir = os.path.join(img_root, cls_name)
        cls_npy_dir = os.path.join(npy_root, cls_name)
        os.makedirs(cls_img_dir, exist_ok=True)
        os.makedirs(cls_npy_dir, exist_ok=True)

        wav_list = [f for f in os.listdir(cls_wav_dir) if f.lower().endswith(".wav")]
        print(f"----- 类别【{cls_name}】，共 {len(wav_list)} 条wav -----")

        for wav_name in wav_list:
            wav_full_path = os.path.join(cls_wav_dir, wav_name)
            base = os.path.splitext(wav_name)[0]

            mel_db = process_single_wav(wav_full_path)

            # 保存npy特征
            npy_save_path = os.path.join(cls_npy_dir, f"{base}.npy")
            np.save(npy_save_path, mel_db)

            # 保存频谱图片
            plt.figure(figsize=(8,3))
            librosa.display.specshow(mel_db, sr=SR, x_axis="time", y_axis="mel", fmax=FMAX)
            plt.colorbar(format="%+2.0f dB")
            plt.title(f"[{cls_name}] {wav_name}")
            plt.tight_layout()
            img_save_path = os.path.join(cls_img_dir, f"{base}.png")
            plt.savefig(img_save_path, dpi=150)
            plt.close()

            total_count +=1
            print(f"✅ {wav_name} | shape={mel_db.shape}")
        print()

    print(f"\n🎉全部处理完成！总共处理 {total_count} 条音频")
    print(f"图片目录：{img_root}")
    print(f"npy特征目录：{npy_root}")
    print(f"类别清单：{cls_map_path}")


if __name__ == "__main__":
    main()

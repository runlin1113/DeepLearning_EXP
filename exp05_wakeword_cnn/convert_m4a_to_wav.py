import os
import subprocess
import shutil

# ===================== 配置区域 =====================
# 路径全部基于本脚本所在目录，自动适配任意机器（不再写死学校电脑路径）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, "dataset")

# 需要转换的唤醒词类别（与 dataset 下的子文件夹名一致）
# 每个类别的【原始录音】统一放在 <类别名>/raw/ 子文件夹内（.m4a）
# 转换后输出到 <类别名>/wav_out/<类别名>_NNN.wav
# 注意：other 为"非唤醒词"负类，用于让模型区分唤醒词与无关语音
CATEGORIES = ["laicai", "qidong", "other"]

# ffmpeg 路径：优先用环境变量，否则在常见位置里找，最后回退到 PATH
def find_ffmpeg():
    env = os.environ.get("FFMPEG_EXE")
    if env and os.path.exists(env):
        return env
    candidates = [
        r"D:\Kinovea\ffmpeg.exe",
        r"C:\Kinovea\ffmpeg.exe",
        r"D:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # 回退到 PATH 中的 ffmpeg
    found = shutil.which("ffmpeg")
    return found

FFMPEG_EXE = find_ffmpeg()
if not FFMPEG_EXE:
    raise SystemExit("找不到 ffmpeg：请设置环境变量 FFMPEG_EXE 指向 ffmpeg 可执行文件，或把 ffmpeg 加入 PATH。")
print(f"使用 ffmpeg: {FFMPEG_EXE}")
# ====================================================


def convert_category(cls_name: str):
    cls_dir = os.path.join(DATASET_ROOT, cls_name)
    if not os.path.isdir(cls_dir):
        print(f"⚠️ 类别文件夹不存在，跳过：{cls_dir}")
        return

    # 原始录音统一放在 raw/ 子文件夹内，避免和 wav_out 混在一起
    raw_dir = os.path.join(cls_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    out_dir = os.path.join(cls_dir, "wav_out")
    os.makedirs(out_dir, exist_ok=True)

    # 收集待转换的 m4a（跳过已经按前缀命名好的，避免重复）
    m4a_list = [
        f for f in os.listdir(raw_dir)
        if os.path.isfile(os.path.join(raw_dir, f))
        and f.lower().endswith(".m4a")
        and not f.startswith(cls_name)
    ]
    m4a_list.sort()
    print(f"\n----- 类别【{cls_name}】，共 {len(m4a_list)} 个 m4a 待转码 -----")

    for idx, old_name in enumerate(m4a_list, start=1):
        old_full = os.path.join(raw_dir, old_name)
        wav_name = f"{cls_name}_{idx:03d}.wav"
        wav_full = os.path.join(out_dir, wav_name)

        # 已存在则跳过，支持断点续跑
        if os.path.exists(wav_full):
            print(f"↪️ 已存在，跳过：{wav_name}")
            continue

        cmd = [
            FFMPEG_EXE,
            "-i", old_full,
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", "16000",
            "-y",
            wav_full,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {old_name} → {wav_name}")
        else:
            print(f"❌ 转换失败：{old_name}")
            print(result.stderr[-500:])


if __name__ == "__main__":
    for c in CATEGORIES:
        convert_category(c)
    print("\n🎉 全部转码完成！")

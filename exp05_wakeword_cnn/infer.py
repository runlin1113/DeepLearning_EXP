import os
import sys
import numpy as np
import torch
import librosa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_output", "wakeword_cnn.pt")

SR = 16000
N_MELS = 40
FMAX = 8000
TARGET_TIME_STEPS = 32


def wav_to_mel(wav_path):
    y, _ = librosa.load(wav_path, sr=SR, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, fmax=FMAX)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    curr_t = mel_db.shape[1]
    if curr_t > TARGET_TIME_STEPS:
        mel_db = mel_db[:, :TARGET_TIME_STEPS]
    else:
        mel_db = np.concatenate([mel_db, np.zeros((N_MELS, TARGET_TIME_STEPS - curr_t))], axis=1)
    return mel_db.astype(np.float32)


def load_model():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    from train_cnn import WakeWordCNN
    model = WakeWordCNN(ckpt["num_classes"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict(wav_path):
    model, ckpt = load_model()
    class_names = ckpt["class_names"]
    mean, std = ckpt["mean"], ckpt["std"]
    mel = wav_to_mel(wav_path)
    x = (torch.from_numpy(mel[None, None, ...]) - mean) / std   # (1,1,40,32)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].numpy()
    order = probs.argsort()[::-1]
    return [(class_names[i], float(probs[i])) for i in order]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python infer.py <音频文件.wav>")
        sys.exit(1)
    path = sys.argv[1]
    res = predict(path)
    print(f"文件: {path}")
    for name, p in res:
        print(f"  {name:>10}: {p*100:5.1f}%")

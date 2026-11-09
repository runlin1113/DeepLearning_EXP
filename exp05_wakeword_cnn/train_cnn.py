import os
import glob
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== 配置（路径基于脚本目录，跨机器可跑） =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NPY_ROOT = os.path.join(BASE_DIR, "preprocessed", "mel_npy")
CLASS_FILE = os.path.join(BASE_DIR, "preprocessed", "class_names.txt")
SAVE_DIR = os.path.join(BASE_DIR, "model_output")
os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
N_MELS = 40
TARGET_TIME_STEPS = 32
NUM_EPOCHS = 120
BATCH_SIZE = 8
LR = 1e-3
VAL_RATIO = 0.2
PATIENCE = 20          # 验证集精度连续 N 轮不提升则早停
# SpecAugment 强度（小样本下不宜过强，否则遮掉过多时频信息反而难学）
TIME_MASK = 5
FREQ_MASK = 3
# ==============================================================================


def load_class_names():
    with open(CLASS_FILE, "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


def load_all_data(class_names):
    """读取所有 npy 特征，返回 (data_list, label_list)"""
    xs, ys = [], []
    for label, cls in enumerate(class_names):
        pattern = os.path.join(NPY_ROOT, cls, "*.npy")
        for path in sorted(glob.glob(pattern)):
            arr = np.load(path).astype(np.float32)          # shape (40, 32)
            arr = np.expand_dims(arr, axis=0)               # (1, 40, 32)
            xs.append(arr)
            ys.append(label)
    return xs, ys


def stratified_split(labels, test_size=VAL_RATIO, seed=SEED):
    rng = random.Random(seed)
    train_idx, val_idx = [], []
    for cls in set(labels):
        idxs = [i for i, l in enumerate(labels) if l == cls]
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * test_size)))
        val_idx += idxs[:n_val]
        train_idx += idxs[n_val:]
    return train_idx, val_idx


def compute_stats(xs, idxs):
    stack = np.concatenate([xs[i] for i in idxs], axis=0)   # (N, 40, 32)
    mean = stack.mean()
    std = stack.std() + 1e-8
    return mean, std


def spec_augment(x, num_time=1, num_freq=1, time_mask=TIME_MASK, freq_mask=FREQ_MASK):
    """训练时在线增强：时间遮罩 + 频率遮罩（SpecAugment 简化版）"""
    x = x.clone()
    _, n_mels, n_steps = x.shape
    for _ in range(num_freq):
        f = random.randint(0, freq_mask)
        f0 = random.randint(0, max(0, n_mels - f))
        x[:, f0:f0 + f, :] = 0.0
    for _ in range(num_time):
        t = random.randint(0, time_mask)
        t0 = random.randint(0, max(0, n_steps - t))
        x[:, :, t0:t0 + t] = 0.0
    return x


class WakeWordDataset(Dataset):
    def __init__(self, xs, ys, mean, std, train=False):
        self.xs = xs
        self.ys = ys
        self.mean = mean
        self.std = std
        self.train = train

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        x = (torch.from_numpy(self.xs[i]) - self.mean) / self.std
        if self.train:
            x = spec_augment(x)
        return x, self.ys[i]


class WakeWordCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 5 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (16,20,16)
        x = self.pool(F.relu(self.conv2(x)))   # (32,10,8)
        x = self.pool(F.relu(self.conv3(x)))   # (64,5,4)
        x = x.flatten(1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            p = out.argmax(1)
            correct += (p == y).sum().item()
            total += y.size(0)
            preds.extend(p.cpu().numpy().tolist())
            targets.extend(y.cpu().numpy().tolist())
    return correct / max(1, total), preds, targets


def plot_curves(history, save_path):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="train loss")
    plt.plot(history["val_loss"], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="train acc")
    plt.plot(history["val_acc"], label="val acc")
    plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend(); plt.title("Accuracy")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion(preds, targets, class_names, save_path):
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(targets, preds):
        cm[t][p] += 1
    plt.figure(figsize=(5, 4.5))
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(n), class_names, rotation=30, ha="right")
    plt.yticks(range(n), class_names)
    plt.xlabel("predicted"); plt.ylabel("true"); plt.title("Confusion Matrix (val)")
    for i in range(n):
        for j in range(n):
            plt.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    class_names = load_class_names()
    num_classes = len(class_names)
    print(f"类别: {class_names}")

    xs, ys = load_all_data(class_names)
    print(f"样本总数: {len(xs)}")

    train_idx, val_idx = stratified_split(ys)
    train_mean, train_std = compute_stats(xs, train_idx)
    print(f"训练集 {len(train_idx)} 条，验证集 {len(val_idx)} 条")
    print(f"特征归一化均值/标准差: {train_mean:.3f} / {train_std:.3f}")

    train_ds = WakeWordDataset([xs[i] for i in train_idx], [ys[i] for i in train_idx],
                               train_mean, train_std, train=True)
    val_ds = WakeWordDataset([xs[i] for i in val_idx], [ys[i] for i in val_idx],
                             train_mean, train_std, train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = WakeWordCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # 验证精度不再提升时自动降低学习率，帮助后期更稳地收敛
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-5)

    best_val_acc = 0.0
    best_epoch = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    no_improve = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        train_loss = running_loss / len(train_ds)
        train_acc, _, _ = evaluate(model, train_loader, device)
        val_acc, val_preds, val_targets = evaluate(model, val_loader, device)
        # 验证损失
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item() * x.size(0)
        val_loss /= max(1, len(val_ds))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "class_names": class_names,
                "mean": float(train_mean),
                "std": float(train_std),
                "num_classes": num_classes,
            }, os.path.join(SAVE_DIR, "wakeword_cnn.pt"))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"⏹ 早停：验证精度连续 {PATIENCE} 轮未提升")
                break
        scheduler.step(val_acc)

    print(f"\n🎉 训练结束。最佳验证精度 {best_val_acc:.3f}（第 {best_epoch} 轮）")

    # 用最佳模型在验证集上出最终指标 + 混淆矩阵
    ckpt = torch.load(os.path.join(SAVE_DIR, "wakeword_cnn.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state"])
    val_acc, val_preds, val_targets = evaluate(model, val_loader, device)
    plot_curves(history, os.path.join(SAVE_DIR, "training_curves.png"))
    plot_confusion(val_preds, val_targets, class_names,
                   os.path.join(SAVE_DIR, "confusion_matrix.png"))

    print(f"\n最终验证集精度: {val_acc:.3f}")
    print("混淆矩阵（行=真实，列=预测）:")
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(val_targets, val_preds):
        cm[t][p] += 1
    print("        " + " ".join(f"{c:>10}" for c in class_names))
    for i, c in enumerate(class_names):
        print(f"{c:>10}" + " ".join(f"{cm[i][j]:>10}" for j in range(n)))
    print(f"\n模型已保存: {os.path.join(SAVE_DIR, 'wakeword_cnn.pt')}")
    print(f"训练曲线:   {os.path.join(SAVE_DIR, 'training_curves.png')}")
    print(f"混淆矩阵:   {os.path.join(SAVE_DIR, 'confusion_matrix.png')}")


if __name__ == "__main__":
    main()

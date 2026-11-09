# 实验二 · 音频采集与实时波形显示

课程实验的第一站：先把「声音」变成可以送进网络的数字信号，顺便把**采集线程与绘图线程分离**这件事做对——后面实验五的唤醒词识别，用的就是这里产出的 wav。

## 文件

| 文件 | 作用 |
|---|---|
| `exp3.py` | 用 `sounddevice` 录 5 秒单声道音频（44.1 kHz），存成 `record.wav` |
| `exp4.py` | 边录边画：实时显示最近 1024 个采样点的波形 |

## 运行

```bash
pip install sounddevice soundfile numpy matplotlib
python exp3.py     # 生成 record.wav（需要麦克风）
python exp4.py     # 弹窗实时波形
```

`exp4.py` 需要图形界面。服务器 / WSL / 无头环境会因没有音频设备报 `PortAudioError`，这不是代码问题。

## 关键实现

**录音**只有三行：

```python
fs, duration = 44100, 5
audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()                                   # 阻塞直到采满 5 秒
sf.write("record.wav", audio_data, fs)      # (N,1) float32
```

**实时波形的坑在于线程**：`sounddevice` 的回调函数跑在**独立的音频线程**里，如果在回调里直接 `plt` 绘图，音频线程会被渲染阻塞，结果是爆音 + 掉帧。所以这里用 `queue.Queue` 做一层缓冲，回调只负责「把数据丢进队列」：

```python
def callback(indata, frames, time, status):
    try:
        q.put_nowait(indata[:, 0])          # 满了就丢，绝不阻塞音频线程
    except queue.Full:
        pass
```

主线程再用 matplotlib 的定时器每 20 ms 取一次队列并重绘：

```python
timer = fig.canvas.new_timer(interval=20)
timer.add_callback(update_plot, fig)
timer.start()
```

这个「生产者-消费者 + 主线程渲染」的模型，后面在实验七的视频流里又用了一次（检测与绘制分离），是实时视觉/音频系统的通用骨架。

## 参数为什么这么定

| 参数 | 取值 | 理由 |
|---|---|---|
| `fs` | 44.1 kHz | 语音识别常用 16 kHz 就够，但这里保留 CD 采样率，方便后续重采样 |
| `blocksize` | 1024 | 1024/44100 ≈ 23 ms，一块刚好对应一次屏幕刷新，延迟与流畅度的平衡点 |
| 队列容量 | 5 | 约 115 ms 的缓冲，能容忍一次绘图卡顿，又不会让画面明显滞后 |

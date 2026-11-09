import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from queue import Queue

# ========== 解决matplotlib中文显示 ==========
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 负号正常显示

# 参数
fs = 44100
blocksize = 1024
q = Queue(maxsize=5)

# 创建画布
fig, ax = plt.subplots()
x = np.arange(blocksize)
line, = ax.plot(x, np.zeros(blocksize))
ax.set_ylim(-0.5, 0.5)
ax.set_title("实时音频波形")

# 音频回调：子线程，只放数据，不绘图！
def callback(indata, frames, time, status):
    if status:
        print(status)
    try:
        q.put_nowait(indata[:,0])
    except:
        pass

# 定时器回调函数（主线程绘图）
def update_plot(fig):
    while not q.empty():
        data = q.get()
        line.set_ydata(data)
    fig.canvas.draw()
    fig.canvas.flush_events()

# 创建定时器，20ms触发一次update_plot
timer = fig.canvas.new_timer(interval=20)
timer.add_callback(update_plot, fig)
timer.start()

# 音频流
stream = sd.InputStream(samplerate=fs, blocksize=blocksize, channels=1, callback=callback)
with stream:
    plt.show(block=True)

timer.stop()
print("程序结束")

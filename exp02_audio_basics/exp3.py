import sounddevice as sd
import soundfile as sf

fs = 44100  #采样率
duration = 5 #录制5秒

print(f"开始录音 {duration} 秒...")
audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
sf.write("record.wav", audio_data, fs)
print("录音完成，保存 record.wav")

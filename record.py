import sounddevice as sd
from scipy.io.wavfile import write

duration = 100
samplerate= 44100
filename= "recorded.wav"
print("Recordong audio via Airpods")
audio = sd.rec(int(duration*samplerate), samplerate=samplerate,channels=1,dtype='int16')
sd.wait()
write(filename, samplerate,audio)
print(f"Recordingf saved to{filename}")
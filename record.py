import sounddevice as sd
import soundfile as sf

# Set your desired parameters
fs = 48000  # Sample rate (Hz)
seconds =  30 # Duration of recording

print("Recording started...")
recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='int16')
sd.wait()
sf.write('voice_sample.wav', recording, fs)
print("Recording complete and saved as 'voice_sample.wav'")

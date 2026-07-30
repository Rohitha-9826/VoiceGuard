from preprocessing.audio_loader import AudioLoader

loader = AudioLoader()

audio, sr = loader.load_audio("sample.wav")

print("Sample Rate :", sr)
print("Shape :", audio.shape)
print("Duration :", len(audio) / sr, "seconds")
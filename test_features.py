from preprocessing.audio_loader import AudioLoader
from preprocessing.feature_extractor import FeatureExtractor

loader = AudioLoader()
extractor = FeatureExtractor()

audio, sr = loader.load_audio("sample.wav")

mfcc = extractor.extract_mfcc(audio)
mel = extractor.extract_mel_spectrogram(audio)

print("MFCC Shape :", mfcc.shape)
print("Mel Spectrogram Shape :", mel.shape)
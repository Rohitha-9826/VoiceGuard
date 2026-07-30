import librosa
import numpy as np

from config import SAMPLE_RATE, N_MFCC, N_MELS


class FeatureExtractor:
    """
    Extracts audio features for deepfake detection.
    """

    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

    def extract_mfcc(self, audio):
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc
        )

        return mfcc

    def extract_mel_spectrogram(self, audio):
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels
        )

        return mel
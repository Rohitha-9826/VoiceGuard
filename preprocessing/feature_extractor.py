import librosa
import numpy as np

from config import SAMPLE_RATE, N_MFCC, N_MELS, MAX_FRAMES


class FeatureExtractor:
    """
    Extracts audio features for deepfake detection.
    """

    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_mels=N_MELS,
        max_frames=MAX_FRAMES,
    ):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.max_frames = max_frames

    def pad_or_trim(self, feature):
        """
        Make every feature matrix have the same width.
        """

        current_frames = feature.shape[1]

        if current_frames < self.max_frames:

            pad_width = self.max_frames - current_frames

            feature = np.pad(
                feature,
                pad_width=((0, 0), (0, pad_width)),
                mode="constant"
            )

        elif current_frames > self.max_frames:

            feature = feature[:, :self.max_frames]

        return feature

    def extract_mfcc(self, audio):

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc
        )

        mfcc = self.pad_or_trim(mfcc)

        return mfcc

    def extract_mel_spectrogram(self, audio):

        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels
        )

        mel = self.pad_or_trim(mel)

        return mel
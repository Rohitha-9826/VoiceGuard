import librosa
import numpy as np
from pathlib import Path

from config import SAMPLE_RATE


class AudioLoader:
    """
    Loads audio files for the VoiceGuard project.
    """

    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate

    def load_audio(self, file_path):
        """
        Load an audio file.

        Parameters
        ----------
        file_path : str or Path
            Path to the audio file.

        Returns
        -------
        audio : numpy.ndarray
            Audio waveform.
        sample_rate : int
            Sampling rate.
        """

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} not found.")

        audio, sample_rate = librosa.load(
            file_path,
            sr=self.sample_rate,
            mono=True
        )

        return audio, sample_rate
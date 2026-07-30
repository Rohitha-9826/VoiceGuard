from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from constants import LABEL_MAP
from preprocessing.audio_loader import AudioLoader
from preprocessing.feature_extractor import FeatureExtractor


class ASVspoofDataset(Dataset):
    """
    PyTorch Dataset for the ASVspoof2019 LA dataset.
    """

    def __init__(self, protocol_path, audio_dir):

        self.protocol_path = Path(protocol_path)
        self.audio_dir = Path(audio_dir)

        self.audio_loader = AudioLoader()
        self.feature_extractor = FeatureExtractor()

        self.metadata = self._load_protocol()

    def _load_protocol(self):

        df = pd.read_csv(
            self.protocol_path,
            sep=r"\s+",
            header=None
        )

        df.columns = [
            "speaker_id",
            "file_name",
            "unused",
            "attack_type",
            "label"
        ]

        df["label"] = df["label"].map(LABEL_MAP)

        return df

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        audio_path = self.audio_dir / f"{row.file_name}.flac"

        # AudioLoader returns (audio, sample_rate)
        audio, _ = self.audio_loader.load_audio(audio_path)

        # Extract MFCC features
        mfcc = self.feature_extractor.extract_mfcc(audio)

        feature = torch.tensor(mfcc, dtype=torch.float32)

        label = torch.tensor(row.label, dtype=torch.long)

        return feature, label
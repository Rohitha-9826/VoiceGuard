import torch
import torch.nn as nn

from models.cnn import CNNFeatureExtractor
from models.bilstm import BiLSTMClassifier


class CNNBiLSTM(nn.Module):
    """
    CNN + BiLSTM Model for Deepfake Audio Detection
    """

    def __init__(self):
        super().__init__()

        self.cnn = CNNFeatureExtractor()

        self.bilstm = BiLSTMClassifier()

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(256, 2)

    def forward(self, x):

        # CNN
        x = self.cnn(x)

        # BiLSTM
        x = self.bilstm(x)

        # Last Time Step
        x = x[:, -1, :]

        # Dropout
        x = self.dropout(x)

        # Classification
        x = self.classifier(x)

        return x
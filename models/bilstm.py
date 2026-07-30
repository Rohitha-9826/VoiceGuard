import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM for sequence learning.
    """

    def __init__(
        self,
        input_size=640,
        hidden_size=128,
        num_layers=2,
        dropout=0.3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

    def forward(self, x):

        # x = (Batch,64,10,100)

        batch = x.size(0)

        x = x.permute(0, 3, 1, 2)

        # (Batch,100,64,10)

        x = x.reshape(batch, 100, 640)

        # (Batch,100,640)

        output, _ = self.lstm(x)

        return output
import torch
import torch.nn as nn


class CNNFeatureExtractor(nn.Module):
    """
    CNN Feature Extractor
    Input:
        (Batch, 40, 400)
    Output:
        (Batch, 64, 10, 100)
    """

    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
        )

    def forward(self, x):

        # (Batch,40,400)

        x = x.unsqueeze(1)

        # (Batch,1,40,400)

        x = self.cnn(x)

        return x
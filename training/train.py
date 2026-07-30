from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.cnn_bilstm import CNNBiLSTM
from training.dataset import ASVspoofDataset
from training.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# Dataset path
if Path("/content/dataset").exists():
    DATASET_ROOT = Path("/content/dataset/LA")
else:
    DATASET_ROOT = Path("dataset/LA")

protocol_path = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"

audio_dir = DATASET_ROOT / "ASVspoof2019_LA_train" / "flac"

dataset = ASVspoofDataset(protocol_path, audio_dir)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

model = CNNBiLSTM().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

trainer = Trainer(
    model,
    optimizer,
    criterion,
    device
)

loss, accuracy = trainer.train_one_epoch(train_loader)

print(f"Training Loss : {loss:.4f}")
print(f"Training Accuracy : {accuracy:.2f}%")
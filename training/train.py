from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.cnn_bilstm import CNNBiLSTM
from training.dataset import ASVspoofDataset
from training.trainer import Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 50)
print("Device:", device)
print("=" * 50)

# Dataset Root
if Path("/content/dataset").exists():
    DATASET_ROOT = Path("/content/dataset/LA")
else:
    DATASET_ROOT = Path("dataset/LA")

# Training Dataset
train_protocol = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"
train_audio = DATASET_ROOT / "ASVspoof2019_LA_train" / "flac"

# Validation Dataset
dev_protocol = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
dev_audio = DATASET_ROOT / "ASVspoof2019_LA_dev" / "flac"

train_dataset = ASVspoofDataset(
    train_protocol,
    train_audio
)

dev_dataset = ASVspoofDataset(
    dev_protocol,
    dev_audio
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

dev_loader = DataLoader(
    dev_dataset,
    batch_size=32,
    shuffle=False,
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

EPOCHS = 5

best_accuracy = 0

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    train_loss, train_acc = trainer.train_one_epoch(train_loader)

    val_loss, val_acc = trainer.validate(dev_loader)

    print(f"Training Loss      : {train_loss:.4f}")
    print(f"Training Accuracy  : {train_acc:.2f}%")

    print(f"Validation Loss    : {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    if val_acc > best_accuracy:

        best_accuracy = val_acc

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print("✅ Best Model Saved!")

print("\nTraining Finished!")
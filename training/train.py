from pathlib import Path

from torch.utils.data import DataLoader

from training.dataset import ASVspoofDataset

# Detect environment (Colab or Local)
if Path("/content/dataset").exists():
    DATASET_ROOT = Path("/content/dataset/LA")
else:
    DATASET_ROOT = Path("dataset/LA")

protocol_path = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.train.trn.txt"

audio_dir = DATASET_ROOT / "ASVspoof2019_LA_train" / "flac"

# Create Dataset
dataset = ASVspoofDataset(
    protocol_path=protocol_path,
    audio_dir=audio_dir
)

# Create DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2
)

print("=" * 50)
print("DataLoader Created Successfully")
print("=" * 50)

print(f"Dataset Size : {len(dataset)}")
print(f"Number of Batches : {len(train_loader)}")

# Get one batch
features, labels = next(iter(train_loader))

print("Feature Batch Shape :", features.shape)
print("Label Batch Shape :", labels.shape)

print("First 10 Labels :", labels[:10])
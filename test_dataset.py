from training.dataset import ASVspoofDataset

protocol_path = "dataset/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"

audio_dir = "dataset/LA/ASVspoof2019_LA_train/flac"

dataset = ASVspoofDataset(
    protocol_path=protocol_path,
    audio_dir=audio_dir
)

print("Dataset Size:", len(dataset))

feature, label = dataset[0]

print("Feature Shape:", feature.shape)
print("Label:", label)
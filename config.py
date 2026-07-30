from pathlib import Path

# Project Root
BASE_DIR = Path(__file__).resolve().parent

# Dataset
DATASET_DIR = BASE_DIR / "dataset"
RAW_DATA_DIR = DATASET_DIR / "raw"
PROCESSED_DATA_DIR = DATASET_DIR / "processed"
METADATA_DIR = DATASET_DIR / "metadata"

# Outputs
OUTPUTS_DIR = BASE_DIR / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

# Audio
SAMPLE_RATE = 16000
N_MFCC = 40
N_MELS = 128

# Training
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20

# Random Seed
RANDOM_SEED = 42
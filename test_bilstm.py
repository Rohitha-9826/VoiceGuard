import torch

from models.bilstm import BiLSTMClassifier

model = BiLSTMClassifier()

x = torch.randn(32, 64, 10, 100)

output = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", output.shape)
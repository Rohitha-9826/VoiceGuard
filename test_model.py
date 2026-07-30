import torch

from models.cnn_bilstm import CNNBiLSTM

model = CNNBiLSTM()

x = torch.randn(32, 40, 400)

output = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", output.shape)
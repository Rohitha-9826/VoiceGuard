import torch
import torch.nn as nn


class Trainer:

    def __init__(self, model, optimizer, criterion, device):

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_one_epoch(self, dataloader):

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in dataloader:

            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(features)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()

            total += labels.size(0)

        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total

        return epoch_loss, accuracy
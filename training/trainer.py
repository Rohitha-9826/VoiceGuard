import torch
from tqdm import tqdm


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

        progress_bar = tqdm(
            dataloader,
            desc="Training",
            leave=False
        )

        for features, labels in progress_bar:

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

            progress_bar.set_postfix(
                loss=loss.item()
            )

        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total

        return epoch_loss, accuracy

    def validate(self, dataloader):

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            progress_bar = tqdm(
                dataloader,
                desc="Validation",
                leave=False
            )

            for features, labels in progress_bar:

                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                predictions = torch.argmax(outputs, dim=1)

                correct += (predictions == labels).sum().item()

                total += labels.size(0)

                progress_bar.set_postfix(
                    loss=loss.item()
                )

        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total

        return epoch_loss, accuracy
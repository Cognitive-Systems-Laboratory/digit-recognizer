import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataloader import DigitDataset


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: str,
                 device: str = "mps",
                 data_dir: str = "./data",
                 file_name: str = "train.csv",
                 batch_size: int = 64):
        self.model = model.to(device)
        self.device = device
        self.configure_optimizer(optimizer=optimizer, model=self.model)
        self.loss_fn = nn.CrossEntropyLoss()

        train_dataset = DigitDataset(data_dir=data_dir, file_name=file_name, mode="train")
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        valid_dataset = DigitDataset(data_dir=data_dir, file_name=file_name, mode="valid", use_augmentation=False)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    def configure_optimizer(self, optimizer: str, model: nn.Module):
        self.optimizer = {
            "adam": optim.Adam,
            "adamw": optim.AdamW
        }[optimizer.lower()](model.parameters())

    def _epoch(self, dataloader: DataLoader, update_params: bool = True):
        for img, y_true in dataloader:
            img, y_true = img.to(self.device), y_true.to(self.device)
            y_pred = self.model(img)

            loss = self.loss_fn(y_pred, y_true)

            if update_params:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def fit(self, epochs=20):
        for e in range(epochs):
            self._epoch(dataloader=self.train_dataloader, update_params=True)
            self._epoch(dataloader=self.valid_dataloader, update_params=False)
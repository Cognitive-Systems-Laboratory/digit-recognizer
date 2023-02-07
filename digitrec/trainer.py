import hydra
import timm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: optim.Optimizer,
                 loaders: dict,
                 loss_fn: nn.Module,
                 device: str = "mps",
                 epochs: int = 20):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.epochs = epochs
        for mode, loader in loaders.items():
            setattr(self, f"{mode}_dataloader", loader)

    def _epoch(self, dataloader: DataLoader, update_params: bool = True):
        for img, y_true in dataloader:
            img, y_true = img.to(self.device), y_true.to(self.device)
            y_pred = self.model(img)

            loss = self.loss_fn(y_pred, y_true)

            if update_params:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def fit(self, epochs: int = None):
        epochs = epochs or self.epochs
        for e in range(epochs):
            self.model.train()
            self._epoch(dataloader=self.train_dataloader, update_params=True)

            self.model.eval()
            self._epoch(dataloader=self.valid_dataloader, update_params=False)
        self.save_checkpoint(model=self.model, model_dir="./ckpt.pt")
        
    def save_checkpoint(self, model: nn.Module, model_dir: str):
        torch.save(model, model_dir)


def setup_trainer(config) -> Trainer:

    model: nn.Module = timm.create_model(**config.model)
    optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())
    device: str = config.device
    loaders = {mode: hydra.utils.instantiate(
                config.data,
                dataset={"mode": mode})
               for mode in config.modes}
    trainer = hydra.utils.instantiate(config.trainer, 
                                      model=model,
                                      optimizer=optimizer,
                                      loaders=loaders,
                                      device=device)
    return trainer
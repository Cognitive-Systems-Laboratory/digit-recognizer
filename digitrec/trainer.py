import hydra
import timm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torchmetrics.functional.classification as tfc
from tqdm import tqdm
import wandb


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

    def _epoch(self, dataloader: DataLoader, epoch: int, update_params: bool = True, mode: str = "train"):
        progress_bar = tqdm(
            dataloader,
            desc=f'{mode.capitalize()} Epoch {epoch}',
            leave=False,
            total=len(dataloader),
        )
        total_loss = 0
        _y_true, _y_pred = [], []
        for idx, (img, y_true) in enumerate(progress_bar):
            img, y_true = img.to(self.device), y_true.to(self.device)
            y_pred = self.model(img)

            loss = self.loss_fn(y_pred, y_true)
            total_loss += loss.item()

            _y_true.append(y_true)
            _y_pred.append(y_pred)

            if update_params:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        _y_true = torch.cat(_y_true, dim=0).cpu().detach()
        _y_pred = torch.cat(_y_pred, dim=0).cpu().detach()
        wandb.log({f"{mode}/loss": total_loss / (idx + 1)}, step=epoch)
        wandb.log({f"{mode}/acc": tfc.multiclass_accuracy(preds=_y_pred, target=_y_true, num_classes=10)}, step=epoch)
        wandb.log({f"{mode}/auroc": tfc.multiclass_auroc(preds=_y_pred, target=_y_true, num_classes=10)}, step=epoch)

    def fit(self, epochs: int = None):
        epochs = epochs or self.epochs
        for e in range(epochs):
            self.model.train()
            self._epoch(dataloader=self.train_dataloader, epoch=e, update_params=True, mode="train")

            self.model.eval()
            self._epoch(dataloader=self.valid_dataloader, epoch=e, update_params=False, mode="valid")
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
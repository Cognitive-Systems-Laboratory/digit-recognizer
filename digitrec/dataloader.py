from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torchvision.transforms as tfm


class DigitDataset(Dataset):

    def __init__(
        self,
        data_dir: Path | str = Path("data/"),
        file_name: str = "train.csv",
        use_augmentation: bool = True,
        mode: str = None,
        test_size: float = 0.2,
        seed: int = 42,
    ) -> None:
        data = pd.read_csv(Path(data_dir) / file_name)
        if "label" in data.columns:
            self.labels: np.ndarray = data["label"].values
        self.digits: np.ndarray = data.filter(like="pixel").values

        self.use_augmentation = use_augmentation
        self.transforms = tfm.Compose([
            tfm.ToTensor(),
            tfm.GaussianBlur(kernel_size=3),
            tfm.RandomRotation(degrees=(-30, 30)),
            tfm.Normalize(mean=0, std=1.),
        ]) if use_augmentation else \
            tfm.Compose([
                tfm.ToTensor(),
                tfm.Normalize(mean=0, std=1.)
            ])
        if mode is not None:
            X_train, X_valid, y_train, y_valid = train_test_split(self.digits, self.labels, test_size=test_size, random_state=seed)
            self.digits, self.labels = {
                "train": (X_train, y_train),
                "valid": (X_valid, y_valid),
            }[mode]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        X = self.digits[index].reshape(28, 28).astype(np.float32)
        X = self.transforms(X)
        if getattr(self, "labels", None) is not None:
            y = self.labels[index]
            return X, y
        else:
            return X

    def __len__(self):
        return len(self.digits)

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as tfm


class DigitDataset(Dataset):
    
    def __init__(
        self,
        data_dir: Path | str = Path("data/"),
        file_name: str = "train.csv",
        use_augmentation: bool = True,
    ) -> None:
        data = pd.read_csv(data_dir / file_name)
        self.labels: np.ndarray = data["label"].values
        self.digits: np.ndarray = data.filter(like="pixel").values

        self.use_augmentation = use_augmentation
        self.transforms = tfm.Compose([
            tfm.ToTensor(),
            tfm.GaussianBlur(kernel_size=3),
            tfm.RandomRotation(degrees=(-30, 30)),
            tfm.RandomCrop(size=(22, 22)),
            tfm.Normalize(mean=0, std=1.),
        ])
        
    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        X = self.digits[index].reshape(28, 28).astype(np.float32)
        if self.use_augmentation:
            X = self.transforms(X)
        y = self.labels[index]
        return X, y

    def __len__(self):
        return len(self.labels)
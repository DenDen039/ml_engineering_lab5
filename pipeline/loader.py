from typing import Any, Dict, Optional, Tuple
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(
        self, dataframe: pd.DataFrame, transform: Optional[Any] = None
    ) -> None:
        self.df = dataframe
        self.transform = transform
        self.transform_default = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.df.iloc[idx]["image"]
        label = self.df.iloc[idx]["label"]

        # Convert image to Tensor
        image = torch.tensor(image, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


def create_data_loader(df: pd.DataFrame, config: Dict[str, Any]) -> DataLoader:
    """
    Create a data loader for a dataset.


    Args:
    - images_dir (str): Directory containing the images.
    - df (pd.DataFrame): DataFrame containing the dataset.
    - config (Dict[str, Any]): Configuration dictionary containing parameters for data loading.

    Returns:
    - DataLoader: DataLoader for the dataset.
    """
    transform: Optional[Any] = config.get("transform", None)
    batch_size: int = config.get("batch_size", 32)
    num_workers: int = config.get("num_workers", 2)

    dataset = ImageDataset(df, transform=transform)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return data_loader

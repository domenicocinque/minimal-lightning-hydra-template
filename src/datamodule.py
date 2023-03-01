from os import path

import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.demos.mnist_datamodule import MNIST
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

DATASETS_PATH = path.join(path.dirname(__file__), "..", "data")


class MyDataModule(LightningDataModule):
    def __init__(self, batch_size: int = 32):
        super().__init__()
        dataset = MNIST(DATASETS_PATH, train=True, download=True, transform=transforms.ToTensor())
        self.mnist_test = MNIST(
            DATASETS_PATH, train=False, download=True, transform=transforms.ToTensor()
        )
        self.mnist_train, self.mnist_val = random_split(dataset, [55000, 5000])
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)

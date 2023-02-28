"""MNIST backbone image classifier example.

To run: python backbone_image_classifier.py --trainer.max_epochs=50
"""
import logging
import pprint
import warnings
from os import path
from typing import Optional

import hydra
import omegaconf
import torch
import wandb
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.demos.mnist_datamodule import MNIST
from pytorch_lightning.loggers import WandbLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

log = logging.getLogger(__name__)

DATASETS_PATH = path.join(path.dirname(__file__), "..", "data")


@hydra.main(config_path="configs/", config_name="default.yaml")
def main(cfg):
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    logger: WandbLogger = hydra.utils.instantiate(cfg.logger)
    datamodule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
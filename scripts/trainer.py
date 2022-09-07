from pytorch_lightning.cli import LightningCLI

from project.models import ZeroShotMaskModel
from project.datasets.keycap import KeycapDataModule

if __name__ == "__main__":
    cli = LightningCLI(ZeroShotMaskModel, KeycapDataModule)

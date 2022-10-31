from dataset import VeRiWildDataset

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path


class VeRiWildDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_path: Path,
        train_txt: str,
        query_txt: str,
        gallery_txt: str,
        train_transform: callable = None,
        test_transform: callable = None,
        batch_size: int = 8,
        num_workers: int = 2,
        pin_memory: bool = False,
    ):
        super().__init__()

        # Files
        self.data_path = data_path
        self.train_txt = train_txt
        self.query_txt = train_txt
        self.gallery_txt = train_txt

        # data transformations
        self.train_transform = train_transform
        self.test_transform = test_transform

        # dataloader data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # datasets
        self.data_train = None
        self.data_query = None
        self.data_gallery = None

    @property
    def num_classes(self):
        return len(self.data_train.vids)

    def setup(self, stage = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_query and not self.data_gallery:
            self.data_train = VeRiWildDataset(
                data_path=self.data_path,
                info_txt=self.train_txt,
                transform=self.train_transform,
                mode='train',
            )
            self.data_query = VeRiWildDataset(
                data_path=self.data_path,
                info_txt=self.query_txt,
                transform=self.test_transform,
                mode='query',
            )
            self.data_gallery = VeRiWildDataset(
                data_path=self.data_path,
                info_txt=self.gallery_txt,
                transform=self.test_transform,
                mode='gallery',
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        query = DataLoader(
            dataset=self.data_query,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        gallery = DataLoader(
            dataset=self.data_gallery,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        return {'query': query, 'gallery': gallery}
from dataset import VeRiWildDataset, get_simple_transform
from datamodule import VeRiWildDataModule
from model_lit import LitClassifier

from pathlib import Path
from torchvision.models import resnet18
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar
import pytorch_lightning as pl


if __name__ == '__main__':
    data_path = Path('/home/parshikov/data/VERI-Wild Dataset')  # /media/mikhail/data/veri_wild_preparation
    data_module = VeRiWildDataModule(
        data_path=data_path,
        train_txt='train_list_start0.txt',
        query_txt='test_3000_id_query.txt',
        gallery_txt='test_3000_id.txt',
        train_transform=get_simple_transform(),
        test_transform=get_simple_transform(),
    )
    data_module.setup()

    model = LitClassifier(
        net=resnet18(num_classes=data_module.num_classes)
    )

    wandb_logger = WandbLogger(project="VeRiWild")
    rich_progress = RichProgressBar()

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        check_val_every_n_epoch=5,
        log_every_n_steps=10,
        max_epochs=50,
        logger=wandb_logger,
        #callbacks=[rich_progress, lr_monitor, checkpoint_callback],
        callbacks=[lr_monitor],
        accelerator="gpu", devices=1
    )

    # Fit
    trainer.fit(model, data_module)

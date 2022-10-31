from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar

# wandb_logger = WandbLogger(project="VeRiWild")
# rich_progress = RichProgressBar()

lr_monitor = LearningRateMonitor(logging_interval='epoch')
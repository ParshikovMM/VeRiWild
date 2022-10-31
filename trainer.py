import pytorch_lightning as pl
from callback import lr_monitor

trainer = pl.Trainer(
    # check_val_every_n_epoch=5,
    log_every_n_steps=10,
    max_epochs=50,
    # logger=wandb_logger,
    # callbacks=[rich_progress, lr_monitor, checkpoint_callback],
    callbacks=[lr_monitor],
    # accelerator="gpu", devices=1
)

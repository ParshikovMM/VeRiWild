import torch
import pytorch_lightning as pl
from typing import Any, List
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torch.optim import AdamW


# define the LightningModule
class LitClassifier(pl.LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
        ):
        super().__init__()

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, pid, camid = batch
        logits = self.forward(x)
        loss = self.criterion(logits, pid)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, pid, camid

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    # Также нужен валидационный цикл

    # Это не трогать
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        sheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=5, max_epochs=self.trainer.max_epochs)
        return [optimizer], [sheduler]
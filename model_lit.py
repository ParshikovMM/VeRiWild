import torch
import pytorch_lightning as pl
import numpy as np
from typing import Any, List
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from torch.optim import AdamW

from validator import compute_cosine_distance, eval_market1501


# define the LightningModule
class LitClassifier(pl.LightningModule):
    def __init__(
            self,
            net: torch.nn.Module):
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

        # Validation
    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        x, pid, camid = batch
        _, emb = self.net(x, return_features=True)

        return {
            'emb': emb, 'pid': pid, 'camid': camid,
        }

    def validation_epoch_end(self, validation_step_outputs):
        query_list, gallery_list = validation_step_outputs

        q_emb = torch.cat([q['emb'] for q in query_list])
        q_pid = torch.cat([q['pid'] for q in query_list]).cpu().numpy().astype(np.int32)
        q_camid = torch.cat([q['camid'] for q in query_list]).cpu().numpy().astype(np.int32)

        g_emb = torch.cat([g['emb'] for g in gallery_list])
        g_pid = torch.cat([g['pid'] for g in gallery_list]).cpu().numpy().astype(np.int32)
        g_camid = torch.cat([g['camid'] for g in gallery_list]).cpu().numpy().astype(np.int32)

        distmat = compute_cosine_distance(q_emb, g_emb)
        cmc, all_AP, all_INP = eval_market1501(distmat, q_pid, g_pid, q_camid, g_camid)

        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)

        rank1 = cmc[0] * 100
        rank5 = cmc[4] * 100
        rank10 = cmc[9] * 100

        self.log("val/mAP", mAP, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/mINP", mINP, on_step=False, on_epoch=True, prog_bar=False)

        self.log("val/Rank1", rank1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/Rank5", rank5, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/Rank10", rank10, on_step=False, on_epoch=True, prog_bar=False)

        # Optimizers
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-3)
        sheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=5, max_epochs=self.trainer.max_epochs)
        # sheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [sheduler]

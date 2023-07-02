# model.py

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from collections import OrderedDict

import hal.models as models
import hal.losses as losses
import hal.metrics as metrics

class Model(pl.LightningModule):
    def __init__(self, opts, dataloader=None):
        super().__init__()
        self.save_hyperparameters(opts)

        if dataloader is not None:
            self.val_dataloader = dataloader.val_dataloader
            self.train_dataloader = dataloader.train_dataloader

        self.circle_model = getattr(models, self.hparams.model_type)(opts)
        self.ellipse_model = getattr(models, self.hparams.model_type)(opts)
        if self.hparams.loss_type is not None:
            self.val_loss = getattr(losses, self.hparams.loss_type)(**self.hparams.loss_options)
        if self.hparams.loss_type is not None:
            self.train_loss = getattr(losses, self.hparams.loss_type)(**self.hparams.loss_options)

        self.acc_trn = getattr(metrics, self.hparams.evaluation_type)(task="binary")
        self.acc_val = getattr(metrics, self.hparams.evaluation_type)(task="binary")
        self.acc_tst = getattr(metrics, self.hparams.evaluation_type)(task="binary")

    def forward(self, x):
        out_c = self.circle_model(x)
        out_e = self.ellipse_model(x)
        out_c = out_c.argmax(dim=-1)
        out_e = out_e.argmax(dim=-1)
        return out_c, out_e

    def on_train_start(self):
        if self.logger is not None:
            if self.global_rank == 0:
                temp = torch.zeros((1, 2)).to(self.device)
                self.logger.experiment.add_graph(self.circle_model, temp)
                temp = []

    def training_step(self, batch, batch_idx):
        input_c, input_e = batch
        points_c, labels_c = input_c[:,:2], input_c[:,2]
        points_e, labels_e = input_e[:,:2], input_e[:,2]
        labels_c = labels_c.long()
        labels_e = labels_e.long()
        out_c = self.circle_model(points_c)
        out_e = self.ellipse_model(points_e)
        loss = self.train_loss(out_c, labels_c) + self.train_loss(out_e, labels_e)
        acc = (self.acc_trn(out_c.argmax(dim=1), labels_c) + self.acc_trn(out_e.argmax(dim=1), labels_e))/2
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True,  prog_bar=True)
        output = OrderedDict({
            'loss': loss,
            'acc': acc
        })
        return output

    def validation_step(self, batch, batch_idx):
        input_c, input_e = batch
        points_c, labels_c = input_c[:, :2], input_c[:, 2]
        points_e, labels_e = input_e[:, :2], input_e[:, 2]
        labels_c = labels_c.long()
        labels_e = labels_e.long()
        out_c = self.circle_model(points_c)
        out_e = self.ellipse_model(points_e)
        loss = self.train_loss(out_c, labels_c) + self.train_loss(out_e, labels_e)
        acc = (self.acc_trn(out_c.argmax(dim=1), labels_c) + self.acc_trn(out_e.argmax(dim=1), labels_e))/2
        self.log('val_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, logger=True,  prog_bar=True)
        output = OrderedDict({
            'loss': loss,
            'acc': acc
        })
        return output

    def testing_step(self, batch, batch_idx):
        input_c, input_e = batch
        points_c, labels_c = input_c[:, :2], input_c[:, 2]
        points_e, labels_e = input_e[:, :2], input_e[:, 2]
        labels_c = labels_c.long()
        labels_e = labels_e.long()
        out_c = self.circle_model(points_c)
        out_e = self.ellipse_model(points_e)
        acc = (self.acc_trn(out_c.argmax(dim=1), labels_c) + self.acc_trn(out_e.argmax(dim=1), labels_e))/2
        self.log('test_acc', acc, on_step=False, on_epoch=True, logger=True,  prog_bar=True)

    def configure_optimizers(self):
        model_parameters = list(self.circle_model.parameters()) + list(self.ellipse_model.parameters())
        optimizer = getattr(torch.optim, self.hparams.optim_method)(
            model_parameters,
            lr=self.hparams.learning_rate, **self.hparams.optim_options)
        if self.hparams.scheduler_method is not None:
            scheduler = getattr(torch.optim.lr_scheduler, self.hparams.scheduler_method)(
                optimizer, **self.hparams.scheduler_options
            )
            return [optimizer], [scheduler]
        else:
            return [optimizer]

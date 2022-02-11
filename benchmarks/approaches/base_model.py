import os
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from torchmetrics.functional.classification import accuracy
from transformers import get_linear_schedule_with_warmup
from benchmarks.utils.soft_targets_loss import SoftTargetsLoss
from benchmarks.utils.training import get_mlp

from pytorch_lightning import Callback
import copy
from time import sleep


class HistoryMetricsCallback(Callback):
    """Callback for logging all the metrics history."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self.metrics.append(each_me)


class BaseModel(pl.LightningModule):
    """This is an implementation of approach called `Base` in the paper. Also, this class
    is a base class for other approaches.
    """
    def __init__(self,
                 backbone_model=None,
                 classifier=None,
                 trainer=None,
                 train_len=0,
                 lr=1e-5,
                 soft_targets=False,
                 **kwargs):
        super().__init__()
        self.save_hyperparameters("train_len", "lr", *kwargs.keys()) # saves to self.hparams
        self.soft_targets = soft_targets
        self.backbone_model = backbone_model
        self.classifier = classifier
        self.trainer = trainer
        self.loss = nn.CrossEntropyLoss() if not self.soft_targets else SoftTargetsLoss()

    @classmethod
    def add_argparse_args(cls, parent_parser, args=None):
        parser = parent_parser.add_argument_group("BaseModel")
        parser.add_argument("--weights_dir", type=str, default=None)
        parser.add_argument("--checkpoint_top_k", type=int, default=1)
        parser.add_argument("--soft_targets", action='store_true')
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument("--dropout", type=float, default=0)
        parser.add_argument("--num_warmup_steps", type=int, default=None)
        return parent_parser

    @classmethod
    def suggest_hparams(cls, trial, benchmark, datamodule):
        """Defines hyper-parameter search space for Optuna"""
        return dict(
            lr=trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        )

    @classmethod
    def trainer_from_args(cls, args, **kwargs):
        logger = pl_loggers.TensorBoardLogger(
            os.path.abspath(args.log_dir),
            name=args.name,
        )

        trainer = pl.Trainer.from_argparse_args(
            args,
            checkpoint_callback=False,
            logger=logger,
            **kwargs,
        )

        checkpoint_dir = os.path.join(
            os.path.abspath(args.checkpoint_root), logger.name, f'version_{logger.version}'
        )
        trainer.callbacks.append(
            ModelCheckpoint(
                monitor='val_acc',
                dirpath=checkpoint_dir,
                filename='{epoch:02d}-{val_acc:.2f}-{train_loss:.2f}',
                save_top_k=args.checkpoint_top_k,
                save_last=True,
                mode="max",
            )
        )

        trainer.callbacks.append(HistoryMetricsCallback())

        trainer.callbacks.append(
            LearningRateMonitor(logging_interval='step')
        )
        return trainer

    @classmethod
    def from_args(cls, args, backbone_cls, datamodule, **kwargs):
        for k, v in kwargs.items():
            setattr(args, k, v)
        backbone_model = backbone_cls.from_args(args, datamodule)
        num_labels = len(datamodule.unique_labels)
        classifier = get_mlp(
            output_dim=num_labels,
            dropout_rate=args.dropout,
        )
        trainer = cls.trainer_from_args(args)
        return cls(
            backbone_model=backbone_model,
            classifier=classifier,
            trainer=trainer,
            train_len=len(datamodule.train),
            **vars(args)
        )

    def load_checkpoint(self, path):
        return self.load_from_checkpoint(
            path,
            trainer=self.trainer,
            backbone_model=self.backbone_model,
            classifier=self.classifier,
            strict=False,
        )

    def load_best_checkopoint(self):
        checkpoint_callbacks = [c for c in self.trainer.callbacks if isinstance(c, ModelCheckpoint)]
        if len(checkpoint_callbacks) != 1:
            raise Exception(
                f'Expected to find one checkpoint callback to load weights from, found: {len(checkpoint_callbacks)}'
                )
        best_model_path = checkpoint_callbacks[0].best_model_path
        
        new_model = self.load_checkpoint(best_model_path)
        self.load_state_dict(new_model.state_dict(), strict=True)
        return self

    def fit(self, datamodule):
        self.trainer.fit(self, datamodule)
        return self.load_best_checkopoint()

    def test(self):
        test_results = self.trainer.test(self)
        return test_results

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        num_training_steps = self.hparams.max_epochs * self.hparams.train_len // self.hparams.batch_size
        num_warmup_steps = self.hparams.num_warmup_steps if self.hparams.num_warmup_steps is not None else num_training_steps//2
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }
        }

    def forward(self, *args, **kwargs):
        embeddings = self.backbone_model(*args, **kwargs)
        logits = self.classifier(embeddings)
        return logits

    def _process_batch(self, batch, batch_idx, **kwargs):
        labels = batch['labels'] if not self.soft_targets else batch['soft_targets']
        logits = self(**batch)
        loss = self.loss(logits, labels)
        return logits, loss

    def training_step(self, batch, batch_idx):
        logits, loss = self._process_batch(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss = self._process_batch(batch, batch_idx)
        acc = accuracy(logits, batch['labels'])
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("hp_metric", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        logits, loss = self._process_batch(batch, batch_idx)
        acc = accuracy(logits, batch['labels'])
        self.log("test_acc", acc, on_step=False, on_epoch=True, logger=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        logits, loss = self._process_batch(batch, batch_idx)
        predictions = torch.argmax(logits, dim=-1)
        return predictions

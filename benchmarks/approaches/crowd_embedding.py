import copy

import numpy as np
import torch
from pytorch_lightning import Callback
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.functional import accuracy

from benchmarks.approaches.base_model import BaseModel
from benchmarks.utils.training import get_mlp
from pytorch_metric_learning import losses


class CrowdEmbeddingConcatModule(nn.Module):
    def __init__(self, n_annotators, annotator_embedding_dim, annotator_dropout=0, init_embeddings=None):
        super().__init__()
        self.annotator_dropout = annotator_dropout
        self.missing_index = nn.Parameter(torch.LongTensor([n_annotators]), requires_grad=False)
        self.embedding = nn.Embedding(n_annotators+1, annotator_embedding_dim)

        if init_embeddings is not None:
            with torch.no_grad():
                self.embedding.weight[:-1, 0] = torch.tensor(init_embeddings)

    def forward(self, outputs, annotators=None):
        if annotators is not None:
            if self.training and self.annotator_dropout:
                drop_idx = np.random.choice(range(len(annotators)), size=int(self.annotator_dropout*len(annotators)))
                annotators[drop_idx] = self.missing_index[0]
            embedded = self.embedding(annotators)
        else:
            missing_embedding = self.embedding(self.missing_index)[0]
            embedded = missing_embedding.repeat(outputs.shape[0], 1)
        classifier_input = torch.concat([F.normalize(outputs), F.normalize(embedded)], dim=1)
        return classifier_input


def batch_to_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device)
    return batch


class PickBestTopk(Callback):
    def __init__(self, *args, top_k_values, datamodule, **kwargs):
        super(PickBestTopk, self).__init__(*args, **kwargs)
        self.top_k_values = top_k_values
        self.datamodule = datamodule

    def on_validation_epoch_start(self, trainer, pl_module):
        metrics = []
        pl_module.eval()
        for k in self.top_k_values:
            pl_module.top_k = k
            preds = []
            with torch.no_grad():
                for batch in self.datamodule.val_dataloader():
                    predictions = pl_module.predict_step(batch_to_device(batch, pl_module.device), None)
                    preds += predictions.cpu().tolist()
            metrics.append(accuracy(torch.tensor(preds), self.datamodule.val.labels))
        best_top_k_idx = np.argmax(metrics)
        best_top_k = self.top_k_values[best_top_k_idx]
        pl_module.top_k = best_top_k

    def on_test_epoch_start(self, trainer, pl_module):
        self.on_validation_epoch_start(trainer, pl_module)


class PretrainEmbeddingsCallback(Callback):
    def __init__(self, *args, n_epochs, **kwargs):
        super(PretrainEmbeddingsCallback, self).__init__(*args, **kwargs)
        self.n_epochs = n_epochs

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch+1 <= self.n_epochs:
            pl_module.backbone_model.freeze()
        else:
            pl_module.backbone_model.unfreeze()


class CrowdEmbedding(BaseModel):
    def __init__(self,
                 crowd_embedding_head=None,
                 annotator_skills=None,
                 top_k=7,
                 top_annotators=None,
                 inference_policy='top_k',
                 **kwargs):
        super().__init__(**kwargs)
        self.crowd_embedding_head = crowd_embedding_head
        self.annotator_skills = annotator_skills
        self.top_k = top_k
        self.top_annotators = top_annotators
        self.inference_policy = inference_policy

        if 'contrastive' in self.hparams and self.hparams.contrastive:
            self.loss_embedding = losses.ContrastiveLoss(
                pos_margin=self.hparams.contrastive_pos_margin,
                neg_margin=self.hparams.contrastive_neg_margin,
            )

    @classmethod
    def add_argparse_args(cls, parent_parser, args=None):
        parent_parser = super(CrowdEmbedding, cls).add_argparse_args(parent_parser, args)
        parser = parent_parser.add_argument_group('CrowdEmbedding')
        parser.add_argument('--annotator_embedding_dim', type=int, default=20)
        parser.add_argument('--annotator_dropout', type=float, default=0)
        parser.add_argument('--inference_policy', type=str, choices=['base', 'top_k'], default='top_k')
        parser.add_argument('--contrastive', action='store_true', default=False)
        parser.add_argument('--contrastive_loss_mult', type=float, default=0.2)
        parser.add_argument('--contrastive_pos_margin', type=float, default=0.1)
        parser.add_argument('--contrastive_neg_margin', type=float, default=0.9)
        parser.add_argument('--enable_init_embeddings', action='store_true', default=False)
        parser.add_argument('--pretrain_embeddings_epochs', type=int, default=0)
        return parent_parser

    @classmethod
    def trainer_from_args(cls, args, datamodule, **kwargs):
        trainer = super(CrowdEmbedding, cls).trainer_from_args(args, **kwargs)
        if args.inference_policy == 'top_k':
            trainer.callbacks.append(
                PickBestTopk(
                    top_k_values=[1, 3, 5, 7, 15, 21],
                    datamodule=datamodule,
                )
            )
        if args.pretrain_embeddings_epochs > 0:
            trainer.callbacks.append(
                PretrainEmbeddingsCallback(
                    n_epochs=args.pretrain_embeddings_epochs,
                )
            )
        return trainer

    @classmethod
    def suggest_hparams(cls, trial, benchmark, datamodule):
        hparams = super(CrowdEmbedding, cls).suggest_hparams(trial, benchmark, datamodule)

        if 'contrastive' in hparams and hparams.contrastive:
            hparams.update(dict(
                contrastive_loss_mult=trial.suggest_float("contrastive_loss_mult", 0, 0.5),
                contrastive_pos_margin=trial.suggest_float("contrastive_pos_margin", 0, 1),
                contrastive_neg_margin=trial.suggest_float("contrastive_neg_margin", 0, 1),
            ))
        return hparams

    @classmethod
    def from_args(cls, args, backbone_cls, datamodule, **kwargs):
        if args.aggregate_train:
            raise ValueError('CrowdEmbedding does not work on aggregated data, try removing --aggregate_train')

        for k, v in kwargs.items():
            setattr(args, k, v)
        trainer = cls.trainer_from_args(args, datamodule=datamodule)
        backbone_model = backbone_cls.from_args(args, datamodule)
        num_labels = len(datamodule.unique_labels)
        classifier = get_mlp(
            output_dim=num_labels,
            dropout_rate=args.dropout,
        )
        n_annotators = len(np.unique(datamodule.train.annotators))

        annotator_skills = datamodule.annotator_skills.fillna(datamodule.annotator_skills.mean())
        normalized_accuracies = np.array((annotator_skills - annotator_skills.mean())/annotator_skills.std())
        crowd_embedding_head = CrowdEmbeddingConcatModule(n_annotators=n_annotators,
                                                          annotator_embedding_dim=args.annotator_embedding_dim,
                                                          annotator_dropout=args.annotator_dropout,
                                                          init_embeddings=normalized_accuracies if args.enable_init_embeddings else None,
                                                          )

        return cls(backbone_model=backbone_model,
                   crowd_embedding_head=crowd_embedding_head,
                   annotator_skills=annotator_skills,
                   classifier=classifier,
                   trainer=trainer,
                   n_annotators=n_annotators,
                   train_len=len(datamodule.train),
                   **vars(args))

    def configure_optimizers(self):
        opt_dict = super().configure_optimizers()
        if not self.hparams.pretrain_embeddings_epochs:
            return opt_dict

        steps_in_epoch = self.hparams.train_len // self.hparams.batch_size

        original_scheduler = opt_dict['lr_scheduler']['scheduler']
        original_lambdas = original_scheduler.lr_lambdas

        new_lambdas = []
        for l in original_lambdas:
            def new_lr_lambda(step):
                if step < (self.hparams.pretrain_embeddings_epochs * steps_in_epoch):
                    return 1
                else:
                    return l(step)
            new_lambdas.append(new_lr_lambda)

        original_scheduler.lr_lambdas = new_lambdas

        return  {
            'optimizer': opt_dict['optimizer'],
            'lr_scheduler': {
                'scheduler': original_scheduler,
                'interval': 'step',
            }
        }

    def get_top_annotators(self):
        if self.top_annotators:
            return self.top_annotators
        return list(self.annotator_skills.sort_values(ascending=False).head(self.top_k).index)

    def fit(self, datamodule):
        self.trainer.fit(self, datamodule)
        return self.load_best_checkopoint()

    def load_checkpoint(self, path):
        return self.load_from_checkpoint(
            path,
            crowd_embedding_head=self.crowd_embedding_head,
            annotator_skills=self.annotator_skills,
            top_annotators=self.top_annotators,
            top_k=self.top_k,
            inference_policy=self.inference_policy,
            backbone_model=self.backbone_model,
            classifier=self.classifier,
            trainer=self.trainer,
            strict=False,
        )

    def training_step(self, batch, batch_idx):
        logits, losses_dict = self._process_batch(batch, batch_idx)
        self.log_dict(losses_dict, on_step=True, on_epoch=True,  prog_bar=True, logger=True)
        self.log("train_loss", losses_dict['loss'], on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return losses_dict

    def _process_batch(self, batch, batch_idx, **kwargs):
        labels = batch['labels']
        logits = self(**batch)
        loss_classifier = self.loss(logits, labels)

        if not self.training:
            return logits, loss_classifier

        if not self.hparams.contrastive:
            return logits, dict(loss=loss_classifier, loss_classifier=loss_classifier)

        annotators_for_contrastive = batch['annotators_triplets']
        annotators_for_contrastive = annotators_for_contrastive[(annotators_for_contrastive != -1).all(dim=1)]

        loss_embedding = 0
        if annotators_for_contrastive.nelement() > 0:
            embeddings = self.crowd_embedding_head.embedding(annotators_for_contrastive.flatten())

            anchors = torch.tensor(list(range(0, len(embeddings), 3)), device=annotators_for_contrastive.device)
            positives = torch.tensor(list(range(1, len(embeddings), 3)), device=annotators_for_contrastive.device)
            negatives = torch.tensor(list(range(2, len(embeddings), 3)), device=annotators_for_contrastive.device)
            contrastive_pair_tuple = (anchors, positives, negatives)
            contrastive_labels = torch.zeros(len(embeddings)) # This is ignored by the loss function because we provide tuple incides
            loss_embedding = self.loss_embedding(embeddings,
                                                 contrastive_labels,
                                                 indices_tuple=contrastive_pair_tuple)
            loss_embedding = self.hparams.contrastive_loss_mult * loss_embedding
        loss = loss_embedding + loss_classifier
        return logits, dict(loss=loss, loss_classifier=loss_classifier, loss_embedding=loss_embedding)

    def forward(self, *args, **kwargs):
        hidden = self.backbone_model(*args, **kwargs)
        if self.training:
            annotators = kwargs.get('annotators')
            embeddings = self.crowd_embedding_head(hidden, annotators)
            logits = self.classifier(embeddings)
            return logits

        if self.inference_policy == 'top_k':
            batch_size = hidden.shape[0]

            top_annotators = self.get_top_annotators()
            hidden = hidden.repeat_interleave(len(top_annotators), 0)
            annotators = []
            for _ in range(batch_size):
                annotators += top_annotators
            annotators = torch.tensor(annotators).to(hidden.device)
            embeddings = self.crowd_embedding_head(hidden, annotators)
            logits = self.classifier(embeddings)

            argmx = torch.argmax(logits, -1)
            new_logits = torch.zeros_like(logits, device=argmx.device).scatter(1, argmx.unsqueeze(1), 1.0).reshape(batch_size, len(top_annotators), -1)
            return new_logits.sum(axis=1)
        elif self.inference_policy == 'base':
            annotators = None
            embeddings = self.crowd_embedding_head(hidden, annotators)
            logits = self.classifier(embeddings)
            return logits

    def validation_step(self, batch, batch_idx):
        loss = super().validation_step(batch, batch_idx)
        self.log("top_k", self.top_k, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return loss

# Adapted from:
# https://github.com/zdchu/CoNAL/blob/main/conal.py
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
from benchmarks.approaches.base_model import BaseModel
from benchmarks.utils.training import batch_identity_matrices, get_mlp
from benchmarks.approaches.differentiable_aggregations import differentiable_ds

class CoNALModule(nn.Module):
    def __identity_init(self, shape):
        out = np.ones(shape) * 0
        if len(shape) == 3:
            for r in range(shape[0]):
                for i in range(shape[1]):
                    out[r, i, i] = 2
        elif len(shape) == 2:
            for i in range(shape[1]):
                out[i, i] = 2
        return torch.Tensor(out)

    def __init__(self,
                 num_annotators,
                 num_class,
                 com_emb_size=20,
                 user_feature=None,
                 ):
        super().__init__()
        self.num_annotators = num_annotators
        self.annotator_confusion_matrices = nn.Parameter(self.__identity_init((num_annotators, num_class, num_class)),
                                   requires_grad=True)

        self.common_confusion_matrix = nn.Parameter(self.__identity_init((num_class, num_class)),
                                          requires_grad=True)

        user_feature = user_feature or np.eye(num_annotators)
        self.user_feature_vec = nn.Parameter(torch.from_numpy(user_feature).float(), requires_grad=False)
        self.diff_linear_1 = nn.LazyLinear(128)
        self.diff_linear_2 = nn.Linear(128, com_emb_size)
        self.user_feature_1 = nn.Linear(self.user_feature_vec.size(1), com_emb_size)

    def simple_common_module(self, input, annotators):
        instance_difficulty = self.diff_linear_1(input)
        instance_difficulty = self.diff_linear_2(instance_difficulty)

        instance_difficulty = F.normalize(instance_difficulty)
        user_feature = self.user_feature_1(self.user_feature_vec[annotators])
        user_feature = F.normalize(user_feature)
        common_rate = torch.sum(instance_difficulty * user_feature, dim=1)
        common_rate = torch.nn.functional.sigmoid(common_rate).unsqueeze(1)
        return common_rate

    def forward(self, embeddings, logits, annotators):
        x = embeddings.view(embeddings.size(0), -1)
        common_rate = self.simple_common_module(x, annotators)
        common_prob = torch.einsum('ij,jk->ik', (F.softmax(logits, dim=-1), self.common_confusion_matrix))
        batch_confusion_matrices = self.annotator_confusion_matrices[annotators]
        indivi_prob = differentiable_ds(logits, batch_confusion_matrices)
        crowd_out = common_rate * common_prob + (1 - common_rate) * indivi_prob  # single instance
        return crowd_out


class CoNAL(BaseModel):
    def __init__(self,
                 conal_head=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.conal_head = conal_head

    @classmethod
    def from_args(cls, args, backbone_cls, datamodule, **kwargs):
        for k, v in kwargs.items():
            setattr(args, k, v)
        backbone_model = backbone_cls.from_args(args, datamodule)

        num_labels = len(datamodule.unique_labels)

        trainer = cls.trainer_from_args(args)
        n_tasks = len(np.unique(datamodule.train.tasks))
        n_annotators = len(np.unique(datamodule.train.annotators))

        classifier = get_mlp(
            output_dim=num_labels,
            dropout_rate=args.dropout,
        )

        conal_head = CoNALModule(
            num_annotators=n_annotators,
            num_class=num_labels,
        )

        return cls(
            backbone_model=backbone_model,
            classifier=classifier,
            conal_head=conal_head,
            trainer=trainer,
            n_tasks=n_tasks,
            n_annotators=n_annotators,
            train_len=len(datamodule.train),
            **vars(args),
        )

    @classmethod
    def add_argparse_args(cls, parent_parser, args=None):
        parent_parser = super(CoNAL, cls).add_argparse_args(parent_parser, args)
        parser = parent_parser.add_argument_group('CoNAL')
        parser.add_argument("--reg_lambda", type=float, default=0.00001)
        return parent_parser

    @classmethod
    def suggest_hparams(cls, trial, benchmark, datamodule):
        hparams = super(CoNAL, cls).suggest_hparams(trial, benchmark, datamodule)
        hparams.update(dict(
            reg_lambda=trial.suggest_float("reg_lambda", 1e-10, 1e-3, log=True),
        ))
        return hparams

    def load_checkpoint(self, path):
        return self.load_from_checkpoint(
            path,
            backbone_model=self.backbone_model,
            classifier=self.classifier,
            conal_head=self.conal_head,
            trainer=self.trainer,
            strict=False,
        )

    def _process_batch(self, batch, batch_idx, **kwargs):
        labels = batch['labels'] if not self.soft_targets else batch['soft_targets']
        logits = self(**batch)
        loss = self.loss(logits, labels)
        if self.training:
            loss -= self.hparams.reg_lambda * torch.sum(
                torch.norm((self.conal_head.annotator_confusion_matrices[batch['annotators']] - self.conal_head.common_confusion_matrix), dim=1, p=2)
            )
        return logits, loss

    def forward(self, *args, **kwargs):
        embeddings = self.backbone_model(*args, **kwargs)
        logits = self.classifier(embeddings)
        if self.training:
            conal_head_output = self.conal_head(embeddings, logits, kwargs['annotators'])
            return conal_head_output
        return logits

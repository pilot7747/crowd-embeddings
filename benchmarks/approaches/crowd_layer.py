import numpy as np
import torch
from torch import nn
from benchmarks.approaches.base_model import BaseModel
from benchmarks.utils.training import batch_identity_matrices, get_mlp


def crowd_layer_mw(outputs, annotators, weight):
    return torch.einsum('lij,ljk->lik', weight[annotators], outputs.unsqueeze(-1)).squeeze()


def crowd_layer_vw(outputs, annotators, weight):
    return weight[annotators] * outputs


def crowd_layer_vb(outputs, annotators, weight):
    return outputs + weight[annotators]


def crowd_layer_vw_b(outputs, annotators, scale, bias):
    return scale[annotators] * outputs + bias[annotators]


class CrowdLayerClassification(nn.Module):
    def __init__(self, num_labels, n_annotators, conn_type='MW', device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CrowdLayerClassification, self).__init__()
        self.conn_type = conn_type

        self.n_annotators = n_annotators
        if conn_type == 'mw':
            self.weight = nn.Parameter(batch_identity_matrices(n_annotators, num_labels, **factory_kwargs))
        elif conn_type == 'vw':
            self.weight = nn.Parameter(torch.ones(n_annotators, num_labels, **factory_kwargs))
        elif conn_type == 'vb':
            self.weight = nn.Parameter(torch.zeros(n_annotators, num_labels, **factory_kwargs))
        elif conn_type == 'vw+b':
            self.scale = nn.Parameter(torch.ones(n_annotators, num_labels, **factory_kwargs))
            self.bias = nn.Parameter(torch.zeros(n_annotators, num_labels, **factory_kwargs))
        else:
            raise ValueError('Unknown connection type for CrowdLayerClassification.')

    def forward(self, outputs, annotators):
        if self.conn_type == 'mw':
            return crowd_layer_mw(outputs, annotators, self.weight)
        elif self.conn_type == 'vw':
            return crowd_layer_vw(outputs, annotators, self.weight)
        elif self.conn_type == 'vb':
            return crowd_layer_vb(outputs, annotators, self.weight)
        elif self.conn_type == 'vw+b':
            return crowd_layer_vw_b(outputs, annotators, self.scale, self.bias)


class CrowdLayer(BaseModel):
    def __init__(self,
                 crowd_layer=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.crowd_layer = crowd_layer

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
        n_annotators = len(np.unique(datamodule.train.annotators))
        crowd_layer = CrowdLayerClassification(num_labels, n_annotators, args.conn_type)
        trainer = cls.trainer_from_args(args)
        return cls(backbone_model=backbone_model,
                   classifier=classifier,
                   crowd_layer=crowd_layer,
                   trainer=trainer,
                   n_annotators=n_annotators,
                   train_len=len(datamodule.train),
                   **vars(args))

    @classmethod
    def add_argparse_args(cls, parent_parser, args=None):
        parent_parser = super(CrowdLayer, cls).add_argparse_args(parent_parser, args)
        parser = parent_parser.add_argument_group('CrowdLayer')
        parser.add_argument('--conn_type', type=str, default='mw', choices=['mw', 'vw', 'vb', 'vw+b'])
        return parent_parser

    def load_checkpoint(self, path):
        return self.load_from_checkpoint(
            path,
            crowd_layer=self.crowd_layer,
            backbone_model=self.backbone_model,
            classifier=self.classifier,
            trainer=self.trainer,
            strict=False,
        )

    def forward(self, *args, **kwargs):
        embeddings = self.backbone_model(*args, **kwargs)
        logits = self.classifier(embeddings)
        if self.training:
            return self.crowd_layer(
                logits,
                kwargs['annotators']
            )
        return logits

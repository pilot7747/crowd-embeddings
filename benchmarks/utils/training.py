import os
import torch
from torch import nn


def batch_identity_matrices(batch_size, dim_size, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    x = torch.eye(dim_size, **factory_kwargs)
    x = x.reshape((1, dim_size, dim_size))
    return x.repeat(batch_size, 1, 1)


def get_last_checkpoint_path(checkpoint_root, experiment_name):
    versions_path = os.path.join(checkpoint_root, experiment_name)
    version_dirs = os.listdir(versions_path)
    last_version_dir = sorted(version_dirs, key=lambda x: int(x.split('version_')[1]))[-1]
    return os.path.join(
        versions_path, last_version_dir, 'last.ckpt'
    )


def filter_transformer_args(kwargs):
    bad_args = {'annotators', 'annotators_triplets', 'tasks', 'soft_targets', 'labels'}
    return {
        key: value for key, value in kwargs.items()
        if key not in bad_args
    }


def get_mlp(output_dim, dropout_rate):
    return nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.LazyLinear(output_dim),
        nn.Tanh(),
        nn.LazyLinear(output_dim),
    )

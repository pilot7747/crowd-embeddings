"""
Adapted from https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2_lstm.ipynb
"""

import torch
from torch import nn
from benchmarks.backbones.base_backbone import BaseBackbone


class LSTMModule(nn.Module):
    def __init__(self,
                 vocab_size,
                 pad_index,
                 embedding_dim=300,
                 hidden_dim=300,
                 n_layers=2,
                 bidirectional=True,
                 dropout=0.1,
                 **kwargs,
                 ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_index)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional,
                            dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, length):
        # ids = [batch size, seq len]
        # length = [batch size]
        embedded = self.dropout(self.embedding(ids))
        # embedded = [batch size, seq len, embedding dim]

        # Lengths required to be on CPU because of
        # https://github.com/pytorch/pytorch/issues/43227
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, length,
                                                            batch_first=True,
                                                            enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        output, output_length = nn.utils.rnn.pad_packed_sequence(packed_output)
        # output = [batch size, seq len, hidden dim * n directions]
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-1], hidden[-2]], dim=-1)
            # hidden = [batch size, hidden dim * 2]
        else:
            hidden = hidden[-1]
            # hidden = [batch size, hidden dim]
        return hidden


class LSTMBackbone(BaseBackbone):
    def forward(self, *args, ids, length, **kwargs):
        embeddings = self.trunk(ids, length)
        return embeddings

    @classmethod
    def add_argparse_args(cls, parent_parser, args=None):
        parent_parser = super(LSTMBackbone, cls).add_argparse_args(parent_parser, args=args)
        parser = parent_parser.add_argument_group("LSTMBackbone")
        parser.add_argument("--embedding_dim", type=int, default=300)
        parser.add_argument("--hidden_dim", type=int, default=300)
        parser.add_argument("--n_layers", type=int, default=2)
        parser.add_argument("--bidirectional", type=bool, default=True)
        return parent_parser

    @classmethod
    def suggest_hparams(cls, trial, benchmark, datamodule):
        return dict(
            dropout=trial.suggest_categorical("dropout", choices=[0, 0.1, 0.2]),
        )

    @classmethod
    def from_args(cls, args, datamodule, **kwargs):
        for k, v in kwargs.items():
            setattr(args, k, v)

        trunk = LSTMModule(
            vocab_size=len(datamodule.train.vocab),
            pad_index=datamodule.train.pad_index,
            **vars(args),
        )

        return cls(trunk=trunk, **vars(args))

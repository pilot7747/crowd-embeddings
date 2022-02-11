import collections
import os
import json
import torch
import torchtext
from crowdkit.metrics.performers import accuracy_on_aggregates
from scipy.special import softmax
from torch.utils.data import Sampler, SequentialSampler
from torchvision import transforms
import numpy as np
import pandas as pd
from torch import nn
from transformers import AutoTokenizer
from PIL import Image
import pickle
from statsmodels.stats.proportion import proportion_confint
from benchmarks.utils.data import get_annotator_triplets, add_additional_fields, get_language_model_tokenizer


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data, labels, train=True, annotators=None, tasks=None, soft_targets=None):
        self.raw_data = raw_data
        self.labels = torch.LongTensor(labels)
        self.train = train
        self.annotators = annotators
        self.tasks = tasks
        self.soft_targets = soft_targets
        self.data = None

        self.annotator_triplets = None
        if annotators:
            self.annotator_triplets = get_annotator_triplets(tasks, labels, annotators)

    def __len__(self):
        return len(self.raw_data)


class TransformerTextDataset(TextDataset):
    def __init__(self,  *args, model_name, weights_dir, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.tokenizer = tokenizer or get_language_model_tokenizer(model_name, weights_dir)
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.data_encodings = self.tokenizer(
            self.raw_data,
            padding="max_length",
            # padding=True,
            truncation=True,
            return_tensors='pt',
        )

    def __getitem__(self, index):
        item_dict = {key: val[index] for key, val in self.data_encodings.items()}
        add_additional_fields(item_dict, self, index)
        return item_dict


class LSTMTextDataset(TextDataset):
    def __init__(self, *args, max_length, min_freq, vocab=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_length = max_length
        self.min_freq = min_freq
        self.tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        tokens = [self.tokenize_text(text) for text in self.raw_data]
        self.lengths = [len(tok) for tok in tokens]

        special_tokens = ['<unk>', '<pad>']
        self.vocab = vocab or torchtext.vocab.build_vocab_from_iterator(
            tokens,
            min_freq=min_freq,
            specials=special_tokens,
        )
        self.unk_index = self.vocab['<unk>']
        self.pad_index = self.vocab['<pad>']
        self.vocab.set_default_index(self.unk_index)

        self.ids = [self.tokens_to_ids(tok) for tok in tokens]

    @classmethod
    def add_argparse_args(cls, parent_parser, args=None):
        parser = parent_parser.add_argument_group("LSTMTextDataset")
        parser.add_argument("--max_length", type=int, default=300)
        parser.add_argument("--min_freq", type=int, default=5)
        return parent_parser

    def tokenize_text(self, text):
        tokens = self.tokenizer(text)[:self.max_length]
        return tokens

    def tokens_to_ids(self, tokens):
        ids = [self.vocab[token] for token in tokens]
        return ids

    def collate(self, batch):
        batch_ids = [i['ids'] for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=self.pad_index, batch_first=True)
        batch_length = [i['length'] for i in batch]
        # Length must be on CPU, so to avoid lightning moving it to gpu, its just a list
        # batch_length = torch.stack(batch_length)
        batch_label = [i['labels'] for i in batch]
        batch_label = torch.stack(batch_label)

        out_batch = {
            'ids': batch_ids,
            'length': batch_length,
            'labels': batch_label,
        }

        if self.annotators is not None:
            out_batch['annotators'] = torch.LongTensor([i['annotators'] for i in batch])
            if self.annotator_triplets is not None:
                triplets = [i['annotators_triplets'] for i in batch]
                out_batch['annotators_triplets'] = torch.stack(triplets)
        if self.tasks is not None:
            out_batch['tasks'] = torch.LongTensor([i['tasks'] for i in batch])
        if self.soft_targets is not None:
            out_batch['soft_targets'] = torch.stack([i['soft_targets'] for i in batch])

        return out_batch

    def __getitem__(self, index):
        item_dict = {
            'length': self.lengths[index],
            'ids': torch.IntTensor(self.ids[index]),
        }
        add_additional_fields(item_dict, self, index)
        return item_dict



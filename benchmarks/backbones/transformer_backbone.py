import json
import os

from transformers import AutoModelForSequenceClassification, AutoConfig

from benchmarks.backbones.base_backbone import BaseBackbone
from benchmarks.utils.training import filter_transformer_args


def patch_tokenizer_file(weights_dir, model_name):
    tokenizer_config_path = os.path.join(weights_dir, model_name, 'tokenizer_config.json')
    with open(tokenizer_config_path) as f:
        tokenizer_config = json.load(f)
    tokenizer_config['tokenizer_file'] = os.path.join(weights_dir, model_name, 'tokenizer.json')
    with open(tokenizer_config_path, 'w') as f:
        json.dump(tokenizer_config, f)


class TransformerBackbone(BaseBackbone):
    def forward(self, *args, **kwargs):
        outputs = self.trunk(*args, **filter_transformer_args(kwargs))
        embeddings = outputs.hidden_states[-1][:, 0, :]
        return embeddings

    @classmethod
    def add_argparse_args(cls, parent_parser, args=None):
        parent_parser = super(TransformerBackbone, cls).add_argparse_args(parent_parser, args=args)
        parser = parent_parser.add_argument_group("TranformerBackbone")
        parser.add_argument("--model_name", type=str, default="roberta-base")
        return parent_parser

    @classmethod
    def from_args(cls, args, datamodule):
        if args.weights_dir:
            patch_tokenizer_file(args.weights_dir, args.model_name)

        config = AutoConfig.from_pretrained(
            os.path.join(args.weights_dir, args.model_name) if args.weights_dir else args.model_name,
            output_hidden_states=True,
        )

        trunk = AutoModelForSequenceClassification.from_config(config)

        return cls(trunk=trunk, **vars(args))

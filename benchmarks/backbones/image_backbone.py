import timm
import os
from benchmarks.backbones.base_backbone import BaseBackbone


class ImageBackbone(BaseBackbone):
    def forward(self, *args, **kwargs):
        embeddings = self.trunk(kwargs['images'])
        return embeddings

    @classmethod
    def add_argparse_args(cls, parent_parser, args=None):
        parent_parser = super(ImageBackbone, cls).add_argparse_args(parent_parser, args=args)
        parser = parent_parser.add_argument_group("ImageBackbone")
        parser.add_argument("--model_name", type=str, default="vgg16")
        parser.add_argument("--load_local_checkpoint", action='store_true')
        return parent_parser

    @classmethod
    def from_args(cls, args, datamodule):
        if not args.load_local_checkpoint:
            trunk = timm.create_model(args.model_name, pretrained=True, global_pool='avg')
        else:
            trunk = timm.create_model(args.model_name, global_pool='avg', checkpoint_path=os.path.join(os.getenv('INPUT_PATH'), f'{args.model_name}.pth'))

        return cls(trunk=trunk, **vars(args))

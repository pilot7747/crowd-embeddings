import pytorch_lightning as pl


class BaseBackbone(pl.LightningModule):
    def __init__(self, trunk, **kwargs):
        super().__init__()
        self.save_hyperparameters(*kwargs.keys())  # saves to self.hparams
        self.trunk = trunk

        if self.hparams.freeze_backbone:
            self.freeze()

    def unfreeze(self, force=False):
        if not self.hparams.freeze_backbone or force:
            super().unfreeze()

    def forward(self, *args, **kwargs):
        embeddings = self.trunk(*args, **kwargs)
        return embeddings

    @classmethod
    def suggest_hparams(cls, trial, benchmark, datamodule):
        return {}

    @classmethod
    def add_argparse_args(cls, parent_parser, args=None):
        parser = parent_parser.add_argument_group("BaseBackbone")
        parser.add_argument("--freeze_backbone", action='store_true', default=False)
        return parent_parser

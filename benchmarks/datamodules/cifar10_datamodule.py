import os

import pandas as pd
from crowdkit.aggregation import MajorityVote, DawidSkene
from torch.utils.data import Subset

from benchmarks.datamodules.base_datamodule import BaseDataModule
from benchmarks.utils.data import (factorize_column, get_soft_targets,
                                   golden_accuracies_lower_conf)
from benchmarks.utils.datasets import ImageDataset, ImageDatasetFromPickle
from torchvision import transforms


class Cifar10DataModule(BaseDataModule):
    task_column = 'image'
    unique_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, *args, img_path='data/cifar10/images', **kwargs):
        super().__init__(*args, **kwargs)

        self.img_path = img_path

    @classmethod
    def from_args(cls, args, **kwargs):
        dataset_class = ImageDataset
        dataset_kwargs = {}

        if args.aggregator and not (args.aggregate_train or args.aggregate_val):
            raise ValueError('Specified aggregator, but aggregate_train=False and aggregate_val=False. '
                             'Specify aggregator=None if you want no aggregation.')

        aggregator = None
        if args.aggregator:
            aggregator_map = {
                'majority_vote': MajorityVote(),
                'dawid_skene': DawidSkene(n_iter=100),
            }
            aggregator = aggregator_map[args.aggregator]

        kwargs_ = vars(args)
        del kwargs_['aggregator']
        kwargs_.update(kwargs)

        return cls(
            aggregator=aggregator,
            dataset_class=dataset_class,
            dataset_kwargs=dataset_kwargs,
            **kwargs_,
        )

    def load_dataframes(self):
        train_df = pd.read_csv(self.train_path).iloc[:self.debug_limit]
        train_df, train_golden_df = self.process_crowd_df(train_df, aggregate=self.aggregate_train)
        val_df = pd.read_csv(self.val_path).iloc[:self.debug_limit]

        val_golden_df = None
        if self.val_crowd:
            val_df, val_golden_df = self.process_crowd_df(val_df, aggregate=self.aggregate_val)
        return train_df, val_df, train_golden_df, val_golden_df

    def get_df_data(self, df):
        img_urls = df[self.task_column].values.tolist()
        img_paths = [os.path.join(self.img_path, url.split('/')[-1]) for url in img_urls]
        labels = df.label.values.tolist()
        label_mapping = pd.Series(range(len(self.unique_labels)), index=self.unique_labels)
        labels = [label_mapping[x] for x in labels]

        soft_targets = get_soft_targets(df, self.unique_labels)
        annotators = factorize_column(df, 'performer')
        tasks = factorize_column(df, 'ASSIGNMENT:task_id')

        return dict(img_paths=img_paths, labels=labels, soft_targets=soft_targets, annotators=annotators, tasks=tasks)

    def make_datasets(self,
                      train_kwargs,
                      val_kwargs,
                      test_kwargs,
                      ):
        train_transforms = transforms.Compose([
            transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_kwargs['transform'] = train_transforms
        val_kwargs['transform'] = test_transforms
        test_kwargs['transform'] = test_transforms

        train = ImageDataset(**train_kwargs)
        val = ImageDataset(**val_kwargs)
        test = ImageDatasetFromPickle(**test_kwargs)

        if self.debug_limit:
            test = Subset(test, indices=list(range(len(test)))[:self.debug_limit])
        return train, val, test

    def setup(self, stage=None):
        train_df, val_df, train_golden_df, val_golden_df = self.load_dataframes()
        train_data = self.get_df_data(train_df)
        val_data = self.get_df_data(val_df)
        test_data = dict(
            path=self.test_path,
        )

        if not self.aggregate_train:
            golden_accuracies = golden_accuracies_lower_conf(train_golden_df)
            self.set_annotator_skills(golden_accuracies, train_golden_df)

        self.train, self.val, self.test = self.make_datasets(
            train_kwargs=train_data,
            val_kwargs=val_data,
            test_kwargs=test_data,
        )

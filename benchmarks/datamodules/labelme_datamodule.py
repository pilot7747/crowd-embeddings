import os

import pandas as pd
from crowdkit.aggregation import MajorityVote, DawidSkene
from torch.utils.data import Subset

from benchmarks.datamodules.base_datamodule import BaseDataModule
from benchmarks.utils.data import (factorize_column, get_soft_targets,
                                   consistency_accuracies)

from torchvision import transforms

from benchmarks.utils.datasets import ImageDataset


class LabelMeDataModule(BaseDataModule):
    task_column = 'image'
    unique_labels = ['coast', 'forest', 'highway', 'insidecity', 'mountain', 'opencountry', 'street', 'tallbuilding']

    def __init__(self, *args,
                 dataset_root_path='data/LabelMe',
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset_root_path = dataset_root_path

    @classmethod
    def add_argparse_args(cls, parent_parser, args=None):
        parser = parent_parser.add_argument_group("LabelMeDataModule")
        parser.add_argument("--dataset_root_path", type=str, default="data/LabelMe")
        parser.add_argument("--debug_limit", type=int, default=None)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--aggregate_train", action='store_true', default=False)
        parser.add_argument("--aggregator", type=str, default=None, choices=[None, 'majority_vote', 'dawid_skene'])
        return parent_parser

    @classmethod
    def from_args(cls, args, **kwargs):
        dataset_class = ImageDataset
        dataset_kwargs = {}

        if args.aggregator and not args.aggregate_train:
            raise ValueError('Specified aggregator, but aggregate_train=False. '
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

    def labelme_text_files_to_toloka_dataframe(self, split):
        answers_fpath = os.path.join(self.dataset_root_path, 'answers.txt')
        filenames_fpath = os.path.join(self.dataset_root_path, f'filenames_{split}.txt')
        labels_fpath = os.path.join(self.dataset_root_path, f'labels_{split}_names.txt')
        labels_numbers_fpath = os.path.join(self.dataset_root_path, f'labels_{split}.txt')

        number_to_str_label = {

        }
        with open(labels_numbers_fpath) as f_labels_numbers, open(labels_fpath) as f_labels:
            for l1, l2 in zip(f_labels_numbers.readlines(), f_labels.readlines()):
                label_num = l1.strip()
                label_str = l2.strip()
                if number_to_str_label.get(label_num) is not None and number_to_str_label.get(label_num) != label_str:
                    raise ValueError(f'Conflicting label str to number mapping. For label {label_num} was {number_to_str_label.get(label_num)}, found: {label_str}')
                number_to_str_label[label_num] = label_str
        rows = []
        if split == 'train':
            with open(answers_fpath) as f_answers, open(filenames_fpath) as f_filenames,  open(labels_fpath) as f_labels:
                for task_id, answers_line in enumerate(f_answers.readlines()):
                    crowd_answers_line = answers_line.strip()
                    filename = f_filenames.readline().strip()
                    gt_label = f_labels.readline().strip()

                    dir = filename.split('_')[0]

                    for annotator_id, crowd_label in enumerate(crowd_answers_line.split(' ')):
                        if crowd_label == '-1':
                            continue
                        record = {
                            'INPUT:image': os.path.join(self.dataset_root_path, split, dir, filename),
                            'OUTPUT:result': number_to_str_label[crowd_label],
                            'ASSIGNMENT:task_id': task_id,
                            'ASSIGNMENT:worker_id': annotator_id,
                            'gt_label': gt_label,
                            'GOLDEN:result': None,
                            'HINT: text': None,
                            'HINT: default_language': None,
                            'ASSIGNMENT:link': None,
                            'ASSIGNMENT:assignment_id': None,
                            'ASSIGNMENT:status': None,
                            'ASSIGNMENT:started': None,
                        }
                        rows.append(record)
        else:
            with open(filenames_fpath) as f_filenames, open(labels_fpath) as f_labels:
                for filenames_line, label_line in zip(f_filenames.readlines(), f_labels.readlines()):
                    filename = filenames_line.strip()
                    gt_label = label_line.strip()

                    dir = filename.split('_')[0]

                    record = {
                        'image': os.path.join(self.dataset_root_path, split, dir, filename),
                        'label': gt_label,
                    }
                    rows.append(record)
        return pd.DataFrame.from_records(rows)

    def load_dataframes(self):
        train_df = self.labelme_text_files_to_toloka_dataframe(split='train').iloc[:self.debug_limit]
        train_df, _ = self.process_crowd_df(train_df, aggregate=self.aggregate_train)
        val_df = self.labelme_text_files_to_toloka_dataframe(split='valid').iloc[:self.debug_limit]
        test_df = self.labelme_text_files_to_toloka_dataframe(split='test').iloc[:self.debug_limit]
        return train_df, val_df, test_df

    def get_df_data(self, df):
        image_paths = df[self.task_column].values.tolist()
        labels = df.label.values.tolist()

        label_mapping = pd.Series(range(len(self.unique_labels)), index=self.unique_labels)
        labels = [label_mapping[x] for x in labels]

        soft_targets = get_soft_targets(df, self.unique_labels)
        annotators = factorize_column(df, 'performer')
        tasks = factorize_column(df, 'ASSIGNMENT:task_id')
        return dict(img_paths=image_paths,
                    labels=labels,
                    soft_targets=soft_targets,
                    annotators=annotators,
                    tasks=tasks)

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
        test = ImageDataset(**test_kwargs)

        return train, val, test

    def setup(self, stage=None):
        train_df, val_df, test_df = self.load_dataframes()
        train_data = self.get_df_data(train_df)
        val_data = self.get_df_data(val_df)
        test_data = self.get_df_data(test_df)

        if not self.aggregate_train:
            skills = consistency_accuracies(train_df)
            self.set_annotator_skills(skills, train_df)

        self.train, self.val, self.test = self.make_datasets(
            train_kwargs=train_data,
            val_kwargs=val_data,
            test_kwargs=test_data,
        )

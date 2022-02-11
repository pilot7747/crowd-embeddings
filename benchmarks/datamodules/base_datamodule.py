from torch.utils.data import DataLoader
import pytorch_lightning as pl
import os
import pandas as pd
from crowdkit.aggregation import MajorityVote
from crowdkit.aggregation import DawidSkene
from benchmarks.utils.data import (factorize_column, get_soft_targets, golden_accuracies_lower_conf)



class BaseDataModule(pl.LightningDataModule):
    task_column = None

    def __init__(
        self,
        aggregator,
        *args,
        train_path='data/imdb/train.csv',
        val_path='data/imdb/val.csv',
        test_path='data/imdb/test.csv',
        batch_size=32,
        num_workers=4,
        debug_limit=None,
        dataset_class=None,
        dataset_kwargs=None,
        aggregate_train=False,
        aggregate_val=False,
        val_crowd=False,
        annotator_skills=None,
        **kwargs,
    ):
        super().__init__()
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug_limit = debug_limit
        self.dataset_class = dataset_class
        self.dataset_kwargs = dataset_kwargs

        self.train = None
        self.val = None
        self.test = None

        self.annotator_skills = annotator_skills

        self.aggregator = aggregator
        self.aggregate_train = aggregate_train
        self.aggregate_val = aggregate_val
        self.val_crowd = val_crowd

    @classmethod
    def add_argparse_args(cls, parent_parser, args=None):
        parser = parent_parser.add_argument_group("BaseDataModule")
        parser.add_argument("--train_path", type=str, default="data/imdb/train.csv")
        parser.add_argument("--val_path", type=str, default="data/imdb/val.csv")
        parser.add_argument("--test_path", type=str, default="data/imdb/test.csv")
        parser.add_argument("--debug_limit", type=int, default=None)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=4)
        parser.add_argument("--aggregate_train", action='store_true', default=False)
        parser.add_argument("--aggregate_val", action='store_true', default=False)
        parser.add_argument("--aggregator", type=str, default=None, choices=[None, 'majority_vote', 'dawid_skene'])
        parser.add_argument("--val_crowd", action='store_true', default=False)
        return parent_parser

    def aggregate_answers(self, answers_df, name, cache=True):
        dir_path = os.path.dirname(self.train_path)
        aggregated_dir_path = os.path.join(dir_path, 'aggregated')
        fname = f'{name}_{self.aggregator.__class__.__name__}.csv'
        fpath = os.path.join(aggregated_dir_path, fname)

        if cache and os.path.exists(fpath):
            return pd.read_csv(fpath)

        aggregated_df = self.aggregator.fit_predict(answers_df)
        aggregated_df = aggregated_df.reset_index()
        aggregated_df.columns = [self.task_column, 'label']
        proba_df = self.aggregator.fit_predict_proba(answers_df)
        aggregated_df = aggregated_df.join(proba_df, on=self.task_column)

        if cache:
            os.makedirs(aggregated_dir_path, exist_ok=True)
            aggregated_df.to_csv(fpath, index=False)
        return aggregated_df

    def process_crowd_df(self, df, aggregate=False):
        df = df.rename(
            {
                f'INPUT:{self.task_column}': self.task_column,
                'OUTPUT:result': 'label',
                'ASSIGNMENT:worker_id': 'performer',
            }, axis=1)
        golden_df = df[~pd.isnull(df['GOLDEN:result'])]
        df = df[pd.isnull(df['GOLDEN:result'])]

        if aggregate:
            name = os.path.splitext(os.path.basename(self.train_path))[0]
            df = self.aggregate_answers(df.rename({self.task_column: 'task'}, axis=1), name=name, cache=True)
            df = df.rename({'task': self.task_column}, axis=1)

        return df, golden_df

    def load_dataframes(self):
        train_df = pd.read_csv(self.train_path).iloc[:self.debug_limit]
        val_df = pd.read_csv(self.val_path).iloc[:self.debug_limit]
        test_df = pd.read_csv(self.test_path).iloc[:self.debug_limit]

        train_df, train_golden_df = self.process_crowd_df(train_df, aggregate=self.aggregate_train)

        val_golden_df = None
        if self.val_crowd:
            val_df, val_golden_df = self.process_crowd_df(val_df, aggregate=self.aggregate_val)

        return train_df, val_df, test_df, train_golden_df, val_golden_df

    def get_df_data(self, df):
        raise NotImplementedError()

    def make_datasets(self,
                      train_kwargs,
                      val_kwargs,
                      test_kwargs,
                      ):
        raise NotImplementedError()

    def set_annotator_skills(self, annotator_skills, train_df):
        for performer in train_df['performer'].unique():
            assert performer in annotator_skills, f'No skill found for worker {performer}, please provide it'

        _, annotators_map = factorize_column(train_df, 'performer', return_mapping=True)

        self.annotator_skills = pd.Series(
            {
                annotators_map[annotator]: skill
                for annotator, skill in annotator_skills.iteritems()
            }
        ).sort_values(ascending=False)

    def setup(self, stage=None):
        train_df, val_df, test_df, train_golden_df, val_golden_df = self.load_dataframes()

        if not self.aggregate_train:
            golden_accuracies = golden_accuracies_lower_conf(train_golden_df)
            self.set_annotator_skills(golden_accuracies, train_golden_df)

        train_data = self.get_df_data(train_df)
        val_data = self.get_df_data(val_df)
        test_data = self.get_df_data(test_df)

        self.train, self.val, self.test = self.make_datasets(
            train_kwargs=train_data,
            val_kwargs=val_data,
            test_kwargs=test_data,
        )

    def train_dataloader(self):
        collate = getattr(self.train, 'collate', None)

        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=collate,
        )

    def val_dataloader(self):
        collate = getattr(self.val, 'collate', None)
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate)

    def test_dataloader(self):
        collate = getattr(self.test, 'collate', None)
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=collate)

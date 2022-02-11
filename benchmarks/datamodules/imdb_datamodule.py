import pandas as pd
from crowdkit.aggregation import MajorityVote
from crowdkit.aggregation import DawidSkene

from benchmarks.datamodules.base_datamodule import BaseDataModule
from benchmarks.utils.data import factorize_column, get_soft_targets
from benchmarks.utils.datasets import TransformerTextDataset, LSTMTextDataset


class IMDbDataModule(BaseDataModule):
    task_column = 'text'
    unique_labels = ['neg', 'pos']

    @classmethod
    def from_args(cls, args, **kwargs):
        dataset_kwargs = {}
        if args.backbone == 'transformer':
            dataset_class = TransformerTextDataset
            dataset_kwargs['model_name'] = args.model_name
            dataset_kwargs['weights_dir'] = args.weights_dir
        elif args.backbone == 'lstm':
            dataset_class = LSTMTextDataset
            dataset_kwargs['max_length'] = args.max_length
            dataset_kwargs['min_freq'] = args.min_freq
        else:
            raise ValueError(f'Invalid backbone for IMDbDataModule: {args.backbone}')

        if args.aggregator and not (args.aggregate_train or args.aggregate_val):
            raise ValueError('Specified aggregator, but aggregate_train=False and aggregate_val=False. '
                             'Specify aggregator=None if you want no aggregation.')

        if args.aggregate_val and not args.val_crowd:
            raise ValueError('Specified --aggregate_val, but clean validation is used. '
                             'Pass --val_crowd to use crowd labels on validation,')

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

    def get_df_data(self, df):
        raw_data = df[self.task_column].values.tolist()
        labels = df.label.values.tolist()
        label_mapping = pd.Series(range(len(self.unique_labels)), index=self.unique_labels)
        labels = [label_mapping[x] for x in labels]

        soft_targets = get_soft_targets(df, self.unique_labels)
        annotators = factorize_column(df, 'performer')
        tasks = factorize_column(df, 'ASSIGNMENT:task_id')
        return dict(raw_data=raw_data, labels=labels, soft_targets=soft_targets, annotators=annotators, tasks=tasks)

    def make_datasets(self,
                      train_kwargs,
                      val_kwargs,
                      test_kwargs,
                      ):
        train_kwargs.update(self.dataset_kwargs)
        val_kwargs.update(self.dataset_kwargs)
        test_kwargs.update(self.dataset_kwargs)

        train = self.dataset_class(**train_kwargs)
        if self.dataset_class == LSTMTextDataset:
            val_kwargs['vocab'] = train.vocab
            test_kwargs['vocab'] = train.vocab
        val = self.dataset_class(**val_kwargs)
        test = self.dataset_class(**test_kwargs)
        return train, val, test

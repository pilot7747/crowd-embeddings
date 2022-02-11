import copy
import pickle
import os
import shutil
from distutils.dir_util import copy_tree

import optuna
from optuna.exceptions import OptunaError

from optuna.integration import PyTorchLightningPruningCallback
from optuna.samplers import TPESampler

from benchmarks.datamodules.imdb_datamodule import IMDbDataModule
from benchmarks.datamodules.cifar10_datamodule import Cifar10DataModule
from benchmarks.backbones import LSTMBackbone, TransformerBackbone, ImageBackbone
from benchmarks.approaches import BaseModel, CoTeaching, CrowdLayer, DiffAgg, CrowdEmbedding, CoNAL
from benchmarks.datamodules.labelme_datamodule import LabelMeDataModule
from benchmarks.utils.datasets import LSTMTextDataset
from benchmarks.utils.training import get_last_checkpoint_path
from time import sleep
from benchmarks.approaches.base_model import HistoryMetricsCallback


def is_main_process():
    return os.getenv('LOCAL_RANK') == '0' or os.getenv('LOCAL_RANK') is None


def dummy_forward_pass(model, datamodule):
    for batch in datamodule.train_dataloader():
        model(**batch)
        break


backbone_classes = dict(
    transformer=TransformerBackbone,
    lstm=LSTMBackbone,
    cv_model=ImageBackbone,
)

approach_classes = dict(
    base=BaseModel,
    crowd_layer=CrowdLayer,
    diff_agg=DiffAgg,
    coteaching=CoTeaching,
    crowd_embedding=CrowdEmbedding,
    conal=CoNAL,
)

datamodule_classes = dict(
    imdb=IMDbDataModule,
    cifar10=Cifar10DataModule,
    labelme=LabelMeDataModule,
)


class Benchmark:
    def __init__(
        self,
        args,
        backbone_cls=None,
        approach_cls=None,
        datamodule_cls=None,
        best_hparams=None,
        results=None,
    ):
        self.args = args
        self.backbone_cls = backbone_cls
        self.approach_cls = approach_cls
        self.datamodule_cls = datamodule_cls
        self.best_hparams = best_hparams
        self.results = results or []

    @property
    def checkpoint_dir(self):
        path = os.path.join(os.path.abspath(self.args.checkpoint_root), self.args.name)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def last_reproduction(self):
        return len(self.results)-1 if self.results else None

    @staticmethod
    def add_argparse_args(parent_parser, args=None):
        parser = parent_parser.add_argument_group("Benchmark")
        parser.add_argument("--dataset", type=str, choices=list(datamodule_classes.keys()))
        parser.add_argument("--name", type=str, default='debug_run')
        parser.add_argument("--log_dir", type=str, default='logs')
        parser.add_argument("--reproduction_iters", type=int, default=1)
        parser.add_argument("--resume_benchmark", action='store_true', default=False)
        parser.add_argument("--checkpoint_root", type=str, default='checkpoints')
        parser.add_argument("--tune_iters", type=int, default=5)
        parser.add_argument("--backbone", type=str, choices=list(backbone_classes.keys()))
        parser.add_argument("--approach", type=str, choices=list(approach_classes.keys()))
        temp_args, _ = parent_parser.parse_known_args(args=args)

        backbone_classes[temp_args.backbone].add_argparse_args(parser, args)
        approach_classes[temp_args.approach].add_argparse_args(parser, args)

        if temp_args.backbone == 'lstm':
            LSTMTextDataset.add_argparse_args(parser, args)

        dm = datamodule_classes[temp_args.dataset]
        dm.add_argparse_args(parser, args)

        return parent_parser

    @classmethod
    def try_load(cls, args):
        try:
            benchmark = cls.load_state(args)
            if vars(benchmark.args) != vars(args):
                benchmark.args = args
                print('\033[93mWarning! New args are different to loaded from pickle! Switching to new args.\033[0m')
            if benchmark.last_reproduction is not None \
                and benchmark.last_reproduction >= benchmark.args.reproduction_iters - 1:
                print('Benchmark has already been completed, nothing to resume.')

            print(f'Hyperparameter tuning: {"Done" if benchmark.best_hparams else "Not done"}')
            print(
                f'Reproduction: {benchmark.last_reproduction}/{benchmark.args.reproduction_iters - 1}. '
            )
            return benchmark
        except Exception as e:
            print(f'State file does not exist, starting from scratch.')

    @classmethod
    def setup(cls, args):
        if args.resume_benchmark:
            loaded_state = cls.try_load(args)
            if loaded_state:
                return loaded_state

        datamodules = {
            'imdb': IMDbDataModule,
            'cifar10': Cifar10DataModule,
            'labelme': LabelMeDataModule,
        }
        datamodule_cls = datamodules[args.dataset]
        backbone_cls = backbone_classes[args.backbone]
        approach_cls = approach_classes[args.approach]

        benchmark = cls(
            args=args,
            backbone_cls=backbone_cls,
            approach_cls=approach_cls,
            datamodule_cls=datamodule_cls,
        )
        return benchmark

    def suggest_hparams(self, *args, **kwargs):
        backbone_hparams = self.backbone_cls.suggest_hparams(*args, **kwargs)
        approach_hparams = self.approach_cls.suggest_hparams(*args, **kwargs)
        backbone_hparams.update(approach_hparams)
        return backbone_hparams

    def setup_datamodule(self, args):
        return self.datamodule_cls.from_args(args)

    def setup_model(self, args, datamodule):
        model = self.approach_cls.from_args(args, self.backbone_cls, datamodule)
        return model

    @classmethod
    def load_state(cls, args):
        checkpoint_path = os.path.join(os.path.abspath(args.checkpoint_root), args.name, f'{args.name}_state.pkl')
        try:
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            print(f'Benchmark state file "{checkpoint_path}" does not exist, starting from scratch.')

    def save_state(self):
        if is_main_process():
            checkpoint_path = os.path.join(self.checkpoint_dir, f'{self.args.name}_state.pkl')
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(self, f)

    def tune_hyperparameters(self):
        def objective(trial: optuna.trial.Trial) -> float:
            trial_args = None
            model = None
            try:
                trial_args = copy.deepcopy(self.args)
                trial_args.name = f'{trial_args.name}/{trial_args.name}__tuning_{trial.number}'

                datamodule = self.setup_datamodule(trial_args)
                datamodule.setup()

                hparams = self.suggest_hparams(trial, self, datamodule)
                print(f'Starting trial for hparams: {hparams}')
                for k, v in hparams.items():
                    setattr(trial_args, k, v)

                model = self.setup_model(trial_args, datamodule)
                dummy_forward_pass(model, datamodule)

                model.fit(datamodule)
                if model.global_rank == 0:
                    for callback in model.trainer.callbacks:
                        if isinstance(callback, HistoryMetricsCallback):
                            best_metric = max([metrics['val_acc'].item() for metrics in callback.metrics])
                            return best_metric
            except BaseException as e:
                if isinstance(e, OptunaError):
                    raise e
                if isinstance(e, KeyboardInterrupt):
                    raise Exception(str(e))
                raise
            finally:
                if model:
                    if model.global_rank == 0:
                        checkpoint_dir = os.path.join(
                            os.path.abspath(trial_args.checkpoint_root), model.logger.name, f'version_{model.logger.version}'
                        )
                        shutil.rmtree(checkpoint_dir)

        sampler = TPESampler(**TPESampler.hyperopt_parameters())

        checkpoint_path = os.path.join(self.checkpoint_dir, f'{self.args.name}_tune.db')
        if is_main_process():
            storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{checkpoint_path}",
            )

            study = optuna.create_study(
                study_name=self.args.name,
                storage=storage,
                load_if_exists=self.args.resume_benchmark,
                direction="maximize",
                sampler=sampler,
            )

            trials_df = study.trials_dataframe()
            print(trials_df.to_markdown())
            trials_completed = len([t for t in trials_df.state if t in ('COMPLETE', 'PRUNED')]) \
                if 'state' in trials_df.columns else 0
            study.optimize(
                objective,
                n_trials=self.args.tune_iters - trials_completed,
                show_progress_bar=True,
            )
            trial = study.best_trial
            best_params = dict(trial.params)
            study.trials_dataframe().to_csv(os.path.join(self.checkpoint_dir, f'{self.args.name}_tune_results.csv'))
            return best_params
        else:
            storage = optuna.storages.RDBStorage(
                url=f"sqlite:///{checkpoint_path}",
            )

            study = optuna.create_study(
                study_name=self.args.name,
                storage=storage,
                load_if_exists=self.args.resume_benchmark,
                direction="maximize",
                sampler=sampler,
            )

            trials_df = study.trials_dataframe()
            trials_completed = len([t for t in trials_df.state if t in ('COMPLETE', 'PRUNED')]) \
                if 'state' in trials_df.columns else 0
            while self.args.tune_iters - trials_completed != 0:
                storage = optuna.storages.RDBStorage(
                    url=f"sqlite:///{checkpoint_path}",
                )

                study = optuna.create_study(
                    study_name=self.args.name,
                    storage=storage,
                    load_if_exists=self.args.resume_benchmark,
                    direction="maximize",
                    sampler=sampler,
                )
                trials = study.get_trials()
                try:
                    print(trials[-1])
                    objective(trials[-1])
                except:
                    pass
                sleep(10)
                trials_df = study.trials_dataframe()
                trials_completed = len([t for t in trials_df.state if t in ('COMPLETE', 'PRUNED')]) \
                    if 'state' in trials_df.columns else 0
            study = optuna.create_study(
                study_name=self.args.name,
                storage=storage,
                load_if_exists=self.args.resume_benchmark,
                direction="maximize",
                sampler=sampler,
            )
            trial = study.best_trial
            best_params = dict(trial.params)
            return best_params

    def run(self):
        try:
            if self.args.tune_iters:
                if self.best_hparams:
                    print('Hparams have already been optimized, using saved best hparams.')
                else:
                    print('Starting hparam tuning')
                    self.best_hparams = self.tune_hyperparameters()
                    print(f'Best hyperparameters found: {self.best_hparams}')
                    self.save_state()

            for i in range(self.last_reproduction or 0, self.args.reproduction_iters):
                reproduction_args = copy.deepcopy(self.args)
                if self.best_hparams is not None:
                    for k, v in self.best_hparams.items():
                        setattr(reproduction_args, k, v)
                reproduction_args.name = os.path.join(reproduction_args.name, f'{reproduction_args.name}_repr_{i}')

                if self.args.resume_benchmark:
                    try:
                        reproduction_args.resume_from_checkpoint = get_last_checkpoint_path(
                            os.path.abspath(reproduction_args.checkpoint_root),
                            reproduction_args.name,
                        )
                        print(f'Located last checkpoint at {reproduction_args.resume_from_checkpoint}')
                    except Exception as e:
                        print('Error when trying to locate last checkpoint, starting reproduction from scratch')
                        print('The error was:', str(e))

                datamodule = self.setup_datamodule(reproduction_args)
                datamodule.setup()
                reproduction_model = self.setup_model(reproduction_args, datamodule)
                dummy_forward_pass(reproduction_model, datamodule)

                print(f'Fitting reproduction {i} with params\n', reproduction_model.hparams)
                reproduction_model.fit(datamodule)
                test_results = reproduction_model.test()
                if is_main_process():
                    self.results.append(test_results)
                    self.save_state()
                else:
                    sleep(5)
        finally:
            self.save_state()
        return self.results

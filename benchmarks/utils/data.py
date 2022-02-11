import collections
import os
import json
import torch
import torchtext
from crowdkit.aggregation import MajorityVote
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


def patch_tokenizer_path(dir_path, model_name):
    tokenizer_config_path = os.path.join(dir_path, model_name, 'tokenizer_config.json')
    with open(tokenizer_config_path) as json_file:
        config = json.load(json_file)
    config['tokenizer_file'] = os.path.join(dir_path, model_name, 'tokenizer.json')
    with open(tokenizer_config_path, 'w') as json_file:
        json.dump(config, json_file)


def get_language_model_tokenizer(model_name, local_weights_dir=None):
    if local_weights_dir:
        patch_tokenizer_path(local_weights_dir, model_name)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(local_weights_dir, model_name), local_files_only=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer


def factorize_column(df, column, return_mapping=False):
    if column not in df:
        return None

    factorized = pd.factorize(df[column])
    if return_mapping:
        return factorized[0].tolist(), pd.Series(range(len(factorized[1])), index=factorized[1])
    else:
        return factorized[0].tolist()


def get_soft_targets(df, columns):
    for column in columns:
        if column not in df:
            return None

    return df[columns].values.tolist()


def golden_accuracies_lower_conf(df):
    golden = df[~df['GOLDEN:result'].isna()]
    unique_workers = pd.unique(golden['performer'])
    golden['correctness'] = (golden['label'] == golden['GOLDEN:result']).astype(int)
    total = golden.groupby('performer').count()['ASSIGNMENT:task_id']
    correct = golden.groupby('performer').sum()['correctness']

    lower_confs = dict()
    for worker_id in unique_workers:
        lower_confs[worker_id] = proportion_confint(correct[worker_id], total[worker_id], method='beta')[0]
    for worker_id in pd.unique(df['performer']):
        if worker_id not in lower_confs:
            lower_confs[worker_id] = None
    return pd.Series(lower_confs).sort_values(ascending=False)


def consistency_accuracies(df, aggregator=None):
    skills = accuracy_on_aggregates(answers=df.rename(columns={'image': 'task'}), by='performer', aggregator=aggregator or MajorityVote())
    return skills.sort_values(ascending=False)


def get_annotator_triplets(tasks, labels, annotators):
    unique_annotators = sorted(np.unique(annotators))

    task_answers = collections.defaultdict(dict)
    for task, label, annotator in zip(tasks, labels, annotators):
        task_answers[task][annotator] = label

    annotator_agreement_matrix = np.zeros((len(unique_annotators), len(unique_annotators)))
    annotator_disagreement_matrix = np.zeros((len(unique_annotators), len(unique_annotators)))

    for task, annotator_labels in task_answers.items():
        for p1, l1 in annotator_labels.items():
            for p2, l2 in annotator_labels.items():
                if p1 == p2:
                    continue
                if l1 == l2:
                    annotator_agreement_matrix[p1, p2] += 1
                elif l1 != l2:
                    annotator_disagreement_matrix[p1, p2] += 1

    annotator_triplets = {}
    for p in unique_annotators:
        distrib_positive = annotator_agreement_matrix[p]
        idx_nonzero = np.nonzero(distrib_positive)[0].tolist()
        distrib_positive = (distrib_positive[idx_nonzero]/(distrib_positive[idx_nonzero]+annotator_disagreement_matrix[p][idx_nonzero]))
        if idx_nonzero:
            distrib_positive = softmax(distrib_positive)
        positives = [idx_nonzero, distrib_positive.tolist()]

        distrib_negative = annotator_disagreement_matrix[p]
        idx_nonzero = np.nonzero(distrib_negative)[0].tolist()
        distrib_negative = (distrib_negative[idx_nonzero]/(distrib_negative[idx_nonzero] + annotator_agreement_matrix[p][idx_nonzero]))
        if idx_nonzero:
            distrib_negative = softmax(distrib_negative)
        negatives = [idx_nonzero, distrib_negative.tolist()]
        annotator_triplets[p] = {'pos': positives, 'neg': negatives}

    return annotator_triplets


def sample_triplet(dataset, index):
    triplet_info = dataset.annotator_triplets[dataset.annotators[index]]

    if triplet_info['pos'][0] and triplet_info['neg'][0]:  # can sample triplet
        positive = np.random.choice(triplet_info['pos'][0], size=1, replace=False,
                                    p=triplet_info['pos'][1]).item()
        negative = np.random.choice(triplet_info['neg'][0], size=1, replace=False,
                                    p=triplet_info['neg'][1]).item()
        return [dataset.annotators[index], positive, negative]
    else:
        # Have to do this, so that batch collate works even if no triplet can be sampled
        return [-1, -1, -1]


def add_additional_fields(item_dict, dataset, index):
    item_dict['labels'] = dataset.labels[index]
    if dataset.annotators is not None:
        item_dict['annotators'] = dataset.annotators[index]
        if dataset.annotator_triplets is not None:
            triplet = sample_triplet(dataset, index)
            item_dict['annotators_triplets'] = torch.LongTensor(triplet)
    if dataset.tasks is not None:
        item_dict['tasks'] = dataset.tasks[index]
    if dataset.soft_targets is not None:
        item_dict['soft_targets'] = torch.Tensor(dataset.soft_targets[index])
    return item_dict


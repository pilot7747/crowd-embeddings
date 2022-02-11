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

from benchmarks.utils.data import get_annotator_triplets, add_additional_fields


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, labels, annotators=None, tasks=None, soft_targets=None, transform=None):
        super(ImageDataset, self).__init__()
        self.img_paths = img_paths
        self.annotators = annotators
        self.tasks = tasks
        self.soft_targets = soft_targets
        self.labels = torch.LongTensor(labels)

        self.imgs = {img_path: self.load_image(img_path) for img_path in set(self.img_paths)}
        self.transform = transform

        self.annotator_triplets = None
        if annotators:
            self.annotator_triplets = get_annotator_triplets(tasks, labels, annotators)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = self.imgs[img_path]
        item_dict = {'images': self.transform(img)}
        add_additional_fields(item_dict, self, index)
        return item_dict

    def load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image


class ImageDatasetFromPickle(torch.utils.data.Dataset):
    def __init__(self, path, transform):
        super(ImageDatasetFromPickle, self).__init__()
        data = self.unpickle(path)
        self.imgs = data[b'data'].reshape((len(data[b'data']), 3, 32, 32))
        self.imgs = self.imgs.transpose(0, 2, 3, 1)
        self.transform = transform
        self.labels = torch.LongTensor(data[b'labels'])

    def __getitem__(self, index):
        img = Image.fromarray(self.imgs[index], 'RGB')
        return {'images': self.transform(img), 'labels': self.labels[index]}

    def __len__(self):
        return len(self.imgs)

    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict_ = pickle.load(fo, encoding='bytes')
        return dict_

import datasets

import torch
from torch.utils.data import DataLoader
import utils.transforms as transforms

import pandas as pd

from collections import OrderedDict
import inspect
import json
from os.path import exists


def load_classes(classes_path):
    if not exists(classes_path):
        raise Exception('You must provide path to existing .json file with target classes')

    with open(classes_path, 'r') as f:
        classes = json.load(f, object_pairs_hook=OrderedDict)

    return classes


def collect_dataset_params(data_params, mode):
    if 'common' not in data_params:
        raise Exception('You must add common parameters into hparams.data_params')

    dataset_params = data_params['common'].copy()

    # Override parameters provided in a specified dataset
    if mode == 'train':
        if 'train_params' not in data_params:
            raise Exception('You must add train_params into hparams.data_params')
        override_params = data_params['train_params']
    else:
        if 'valid_params' not in data_params:
            raise Exception('You must add valid_params into hparams.data_params')
        override_params = data_params['valid_params']

    for param_name, param in override_params.items():
        dataset_params[param_name] = param

    return dataset_params


def prepare_transforms(tranform_params):
    transforms_list = []
    for transform_info in tranform_params:
        transform_name = transform_info['name']
        transform_params = transform_info['params']
        if transform_params is not None:
            transform = transforms.__dict__[transform_name](**transform_params)
        else:
            transform = transforms.__dict__[transform_name]()
        transforms_list.append(transform)
    transform = transforms.Compose(transforms_list)

    return transform


def prepare_dataset(data_params, mode):
    # TODO: check required arguments in dataset

    def _validate_dataset_parameters(dataset_class, dataset_params):
        known_params = dict(inspect.getmembers(dataset_class.__init__.__code__))['co_varnames'][1:]

        for param_name in dataset_params.keys():
            if param_name not in known_params:
                raise Exception('Unknown parameter {} for dataset {}'
                                .format(param_name, dataset_name))

    if 'dataset_name' not in data_params:
        raise Exception('You must add dataset_name into hparams.data_params')
    dataset_name = data_params['dataset_name']
    if dataset_name not in datasets.__dict__:
        raise Exception('Unknown dataset {}'.format(dataset_name))
    dataset_class = datasets.__dict__[dataset_name]

    dataset_params = collect_dataset_params(data_params, mode)
    if 'transform' in dataset_params:
        dataset_params['transform'] = prepare_transforms(dataset_params['transform'])
    if 'classes' in dataset_params:
        dataset_params['classes'] = load_classes(dataset_params['classes'])
    if 'labels' in dataset_params:
        labels = pd.read_csv(dataset_params['labels'])
        labels = labels.reset_index()
        dataset_params['labels'] = labels

    _validate_dataset_parameters(dataset_class, dataset_params)
    dataset_ = dataset_class(**dataset_params)

    return dataset_


def prepare_data_loaders(hparams, mode):
    if mode not in ['train', 'valid', 'test']:
        raise Exception("Mode should be in ['train', 'valid', 'test']")

    if 'data_params' not in hparams:
        raise Exception('You must provide data params in hparams')

    data_params = hparams['data_params']
#     if 'common' not in data_params or 'classes' not in data_params['common']:
#         raise Exception('You must add classes into hparams.data_params.common')
#     classes = load_classes(data_params['common']['classes'])

    dataset = prepare_dataset(data_params, mode)

    if 'training_params' not in hparams or 'batch_size' not in hparams['training_params']:
        raise Exception('You must add training_params with batch_size specified in hparams')
    training_params = hparams['training_params']

    n_workers = data_params['n_workers'] if 'n_workers' in data_params else 0

    loader = DataLoader(dataset, batch_size=training_params['batch_size'],
                        shuffle=True, num_workers=n_workers,
                        pin_memory=torch.cuda.is_available())

    return loader

import os
import argparse
import importlib

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import Configuration
import pdb


def instantiate(module, name):
    module = importlib.import_module(module)
    instance = getattr(module, name)
    return instance


def instantiate_metrics(metrics, experiment, split):
    module = importlib.import_module('metrics')
    for key, metric in metrics.items():
        metric = getattr(module, metric)
        metrics.update({key: metric(getattr(experiment, '{}_writer'.format(split)))})
    return metrics


def train(config_file):
    configuration = Configuration(config_file)

    # Instantiate model
    model_module, model_name = configuration.get('model', 'module'), configuration.get('model', 'name')
    model = instantiate(model_module, model_name)

    # Instantiate loss
    loss_module, loss_name = configuration.get('loss', 'module'), configuration.get('loss', 'name')
    loss = instantiate(loss_module, loss_name)

    # Instantiate optimizer
    optimizer_module, optimizer_name = configuration.get('optimizer', 'module'), configuration.get('optimizer', 'name')
    optimizer = instantiate(optimizer_module, optimizer_name)

    # Instantiate experiment
    experiment_module = configuration.get('experiment', 'module')
    experiment_name = configuration.get('experiment', 'name')
    experiment = instantiate(experiment_module, experiment_name)

    # Initialize experiment

    experiment = experiment(model, loss, optimizer, configuration)

    # Instantiate metrics
    train_metrics = instantiate_metrics(configuration.get_section('train metrics'), experiment, 'train')
    val_metrics = None
    if configuration.has_section('val metrics'):
        val_metrics = instantiate_metrics(configuration.get_section('val metrics'), experiment, 'val')

    # Get train dataloader parameters
    transform = transforms.Compose([transforms.ToTensor()])
    batch_size = int(configuration.get('train loader', 'batch size'))
    shuffle = bool(configuration.get('train loader', 'shuffle'))
    num_workers = int(configuration.get('train loader', 'num workers'))
    pin_memory = torch.cuda.is_available()

    # Initialize train dataloader
    train_dataloader_module = configuration.get('train loader', 'module')
    train_dataloader_name = configuration.get('train loader', 'name')
    skip = int(configuration.get('train loader', 'skip'))
    split = configuration.get('train loader', 'split')
    dataset = instantiate(train_dataloader_module, train_dataloader_name)

    dataset = dataset(split=split, skip=skip, flattened=False, transform=transform)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.7*len(dataset)), int(0.3*len(dataset))])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=pin_memory)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory)
    test_dataloader = None
    if configuration.has_section('test dataloader'):
        # Get test dataloader parameters
        file_path = configuration.get('test loader', 'file path')
        transform = None
        batch_size = int(configuration.get('test loader', 'batch size'))
        shuffle = bool(configuration.get('test loader', 'shuffle'))
        num_workers = int(configuration.get('test loader', 'num_workers'))
        pin_memory = torch.cuda.is_available()

        # Initialize train dataloader
        test_dataloader_module = configuration.get('test loader', 'module')
        test_dataloader_name = configuration.get('test loader', 'name')
        test_dataset = instantiate(test_dataloader_module, test_dataloader_name)
        test_dataset = test_dataset(file_path, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                                     num_workers=num_workers, pin_memory=pin_memory)

    experiment.train_and_validate(train_dataloader, val_dataloader, {'train': train_metrics, 'val': val_metrics})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model by providing a configuration file')
    parser.add_argument("--config_file", type=str, help="Path to the configuration file",
                        default=os.path.join('configuration', 'base', 'config.ini'))
    args = parser.parse_args()

    train(args.config_file)

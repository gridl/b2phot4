import os
import argparse
import importlib
from torchvision.transforms import ToTensor
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from utils import Configuration
from processing.loader import HoromaDataset



def train(config_file):
    dataset = HoromaDataset('valid', 1331, use_overlap=True)
    configuration = Configuration(config_file)
    test = 1
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

    train_metrics = instantiate_metrics(configuration.get_section('train metrics'), experiment, 'train')
    val_metrics = instantiate_metrics(configuration.get_section('val metrics'), experiment, 'val')

    experiment.train_and_validate(train, test, {'train': train_metrics, 'val': val_metrics})


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model by providing a configuration file')
    parser.add_argument("--config_file", type=str, help="Path to the configuration file",
                        default=os.path.join('configuration', 'base', 'config.ini'))
    args = parser.parse_args()

    train(args.config_file)

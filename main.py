import os
import argparse
import importlib

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import Configuration
import numpy as np
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
    train_batch_size = int(configuration.get('train loader', 'batch size'))
    train_shuffle = bool(configuration.get('train loader', 'shuffle'))
    train_num_workers = int(configuration.get('train loader', 'num workers'))
    pin_memory = torch.cuda.is_available()

    # Initialize train dataloader
    train_dataloader_module = configuration.get('train loader', 'module')
    train_dataloader_name = configuration.get('train loader', 'name')
    train_skip = int(configuration.get('train loader', 'skip'))
    train_split = configuration.get('train loader', 'split')
    train_dataset = instantiate(train_dataloader_module, train_dataloader_name)

    # Get valid dataloader parameters
    val_batch_size = int(configuration.get('train loader', 'batch size'))
    val_num_workers = int(configuration.get('train loader', 'num workers'))

    # Initialize valid dataloader
    valid_dataloader_module = configuration.get('valid loader', 'module')
    valid_dataloader_name = configuration.get('valid loader', 'name')
    val_skip = int(configuration.get('valid loader', 'skip'))
    val_split = configuration.get('valid loader', 'split')
    valid_dataset = instantiate(valid_dataloader_module, valid_dataloader_name)

    # Obtaining train & valid dataloaders
    train_dataset = train_dataset(split=train_split, skip=train_skip, flattened=False, transform=transform)
    valid_dataset = valid_dataset(split=val_split, skip=val_skip, flattened=False, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=train_shuffle,
                                  num_workers=train_num_workers, pin_memory=pin_memory)
    val_dataloader = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=False,
                                num_workers=val_num_workers, pin_memory=pin_memory)
    test_dataloader = None
    if configuration.has_section('test dataloader'):
        # Get test dataloader parameters
        file_path = configuration.get('test loader', 'file path')
        transform = None
        batch_size = int(configuration.get('test loader', 'batch size'))
        shuffle = bool(configuration.get('test loader', 'shuffle'))
        num_workers = int(configuration.get('test loader', 'num_workers'))
        pin_memory = torch.cuda.is_available()

        # Initialize test dataloader
        test_dataloader_module = configuration.get('test loader', 'module')
        test_dataloader_name = configuration.get('test loader', 'name')
        test_dataset = instantiate(test_dataloader_module, test_dataloader_name)
        test_dataset = test_dataset(file_path, transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle,
                                     num_workers=num_workers, pin_memory=pin_memory)

    #Setting up the environment
    metrics = {'train': train_metrics, 'val': val_metrics}
    cluster = {}
    seed = int(configuration.get('Kmeans parameters', 'seed'))
    k = int(configuration.get('Kmeans parameters', 'nb_clusters'))

    experiment.train_and_validate(train_dataloader,
                                  val_dataloader,
                                  {'train': train_metrics,
                                   'val': val_metrics},
                                  k,
                                  seed)

    # for epoch in range(experiment.parameters.get('init_epoch', 0), int(experiment.parameters.get('num_epochs'))):
    #    # Run one epoch
    #    logging.info("Epoch {}/{}".format(epoch + 1, experiment.parameters.get('num_epochs')))
    #
    #     # Compute number of batches in one epoch (one full pass over the training set)
    #     train_metrics = experiment.train_autoencoder(train_dataloader, metrics.get('train'))
    #
    #     # Register train metrics to tensorboard
    #     experiment.register_metrics(train_metrics, epoch + 1)
    #
    #     # Evaluate for one epoch on validation set
    #     val_metrics = experiment.eval_autoencoder(val_dataloader, metrics.get('val'))
    #
    #     # Register validation metrics to tensorboard
    #     experiment.register_metrics(val_metrics, epoch + 1)
    #
    #     #train and validate kmeans cluster using new data embedding
    #     new_train_dataset = experiment.model.encoder(torch.from_numpy(train_dataset.dataset.data)).numpy()
    #     new_valid_dataset = experiment.model.encoder(torch.from_numpy(valid_dataset.dataset.data)).numpy()
    #
    #     if k is None:
    #         k = len(valid_dataset.dataset.map_labels)
    #     else:
    #         k = k
    #
    #     # Train and eval. One model per task
    #     print("\n### Clustering on Species ###")
    #     cluster['kmeans'] = train_kmeans(new_train_dataset, k, seed)
    #     model['cluster_label'] = assign_labels_to_clusters(cluster['kmeans'], new_train_dataset, train_dataset.dataset.targets)
    #
    #     print("VALID # ", end='')
    #     predictions = eval_kmeans(model, new_valid_dataset, valid_dataset.dataset.targets)
    #
    #     # Saving final model
    #     experiment_name = re.sub(" +", '__', re.sub('[(){}<>,/!?]', '', str(args)[9:]))
    #     saved_model_name = "{}_model.joblib".format(experiment_name)
    #     dump(model, os.path.join(experiments_dir, saved_model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model by providing a configuration file')
    parser.add_argument("--config_file", type=str, help="Path to the configuration file",
                        default=os.path.join('configuration', 'base', 'config.ini'))
    args = parser.parse_args()

    train(args.config_file)

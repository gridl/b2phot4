#!/usr/bin/env python3
import os
import shutil
import logging
from tqdm import tqdm

from utils.misc import set_logger

import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, f1_score
import warnings  # To mute scikit-learn warnings about f1 score.
warnings.filterwarnings("ignore")

import pdb
import time
import numpy as np


class Experiment(object):
    """Base class for experiments"""

    def __init__(self, model, loss, optimizer, configuration, restore_file=None, only_weights=False):
        self.configuration = configuration
        self.experiment_path = configuration.get_path()
        self.model = model()
        self.loss = loss()
        self.optimizer = optimizer(self.model.parameters(), **configuration.get_section('optimizer parameters'))
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.parameters = configuration.get_section('setup')
        self.setup()
        if restore_file is not None:
            self.load_checkpoint(restore_file, only_weights)
        self.train_writer = SummaryWriter(os.path.join(self.experiment_path, 'tensorboard', 'train'))
        self.val_writer = SummaryWriter(os.path.join(self.experiment_path, 'tensorboard', 'val'))

    def compute_loss(self, output_batch, labels_batch):
        return self.loss(output_batch, labels_batch)

    @staticmethod
    def compute_metrics(metrics, output_batch, labels_batch):
        for metric in metrics.values():
            metric(output_batch, labels_batch)

    def load_checkpoint(self, restore_file, only_weights):
        if not os.path.exists(restore_file):
            raise ("File {} doesn't exist ".format(restore_file))

        checkpoint = torch.load(restore_file)
        self.model.load_state_dict(checkpoint['model_state'])
        if not only_weights:
            self.optimizer.load_state_dict(checkpoint.get('optimizer_state'))
            self.parameters.update({'init_epoch': checkpoint.get('epoch', 0)})

        return checkpoint

    def predict(self, dataloader):
        raise NotImplementedError

    @staticmethod
    def register_metrics(metrics, epoch):
        for metric in metrics.values():
            metric.write_to_tensorboard(epoch)

    @staticmethod
    def reset_metrics(metrics):
        for metric in metrics.values():
            metric.reset()

    def save_checkpoint(self, epoch, is_best):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        state_filepath = os.path.join(self.experiment_path, 'models', 'last.pth.tar')
        model_filepath = os.path.join(self.experiment_path, 'models', 'last_model.pth')
        torch.save(state, state_filepath)
        torch.save(self.model, model_filepath)
        if is_best:
            shutil.copyfile(state_filepath, os.path.join(self.experiment_path, 'models', 'best.pth.tar'))
            shutil.copyfile(model_filepath, os.path.join(self.experiment_path, 'models', 'best_model.pth'))

    def setup(self):
        # Setup random seed for the experiment if none provided use 0
        torch.manual_seed(self.parameters.get('random_seed', 0))
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.parameters.get('random_seed', 0))

        # Create directories to store results
        for directory in ['log', 'results', 'models', 'tensorboard']:
            directory_path = os.path.join(self.experiment_path, directory)
            if not os.path.exists(directory_path):
                os.mkdir(directory_path)

        # Set the logger
        set_logger(os.path.join(self.experiment_path, 'log', 'experiment.log'))

    def train_and_validate(self, train_dataloader, val_dataloader, metrics, k, seed):
        for epoch in range(self.parameters.get('init_epoch', 0), int(self.parameters.get('num_epochs'))):
            # Run one epoch
            logging.info("Epoch {}/{}".format(epoch + 1, self.parameters.get('num_epochs')))

            # Compute number of batches in one epoch (one full pass over the training set)
            train_metrics = self.train_autoencoder(train_dataloader, metrics.get('train'))

            # Register train metrics to tensorboard
            self.register_metrics(train_metrics, epoch + 1)

            # Evaluate for one epoch on validation set
            val_metrics = self.eval_autoencoder(val_dataloader, metrics.get('val'))

            # Register validation metrics to tensorboard
            self.register_metrics(val_metrics, epoch + 1)

            # k means stuff
            kmeans = self.train_kmeans(train_dataloader, k, seed)
            cluster_preds = self.eval_kmeans(kmeans, val_dataloader)
            y_preds = self.assign_labels_to_clusters(kmeans, cluster_preds, val_dataloader.dataset.targets)
            pdb.set_trace()
            # val_acc = val_metrics['accuracy'].get_accuracy()
            # is_best = val_acc >= best_val_acc

    def train_autoencoder(self, dataloader, metrics):
        # set model to training mode
        self.model.train()

        # Reset all metrics
        self.reset_metrics(metrics)

        # Use tqdm for progress bar
        with tqdm(total=len(dataloader)) as t:
            for i, (train_batch, labels_batch) in enumerate(dataloader):
                # move to GPU if available
                self.model = self.model.to(self.device)
                train_batch, labels_batch = train_batch.to(self.device), labels_batch.to(self.device)

                # compute model output and loss
                output_batch = self.model(train_batch)
                loss = self.compute_loss(output_batch, train_batch)

                # clear previous gradients, compute gradients of all variables wrt loss
                self.optimizer.zero_grad()
                loss.backward()
                # update the average loss
                metrics['loss'].update(loss.item())

                # performs updates using calculated gradients
                self.optimizer.step()

                # Evaluate summaries only once in a while
                if i % int(self.parameters.get('save_summary_steps')) == 0:
                    # extract data from torch Variable, move to cpu, convert to numpy arrays
                    output_batch = output_batch.data.cpu().numpy()
                    labels_batch = labels_batch.data.cpu().numpy()

                    # compute all metrics on this batch
                    self.compute_metrics(metrics, output_batch, labels_batch)

                t.set_postfix(loss='{:05.3f}'.format(metrics['loss'].get_average()))
                t.update()

        # Summary of metrics in log
        metrics_string = "".join([str(metric) for metric in metrics.values()])
        logging.info("- Train metrics: \n" + metrics_string)
        return metrics

    def eval_autoencoder(self, dataloader, metrics):
        # set model to evaluation mode
        self.model.eval()

        # Reset all metrics
        self.reset_metrics(metrics)

        # compute metrics over the dataset
        for data_batch, labels_batch in dataloader:

            # move to GPU if available
            data_batch, labels_batch = data_batch.to(self.device), labels_batch.to(self.device)

            # compute model output
            output_batch = self.model(data_batch)
            loss = self.compute_loss(output_batch, data_batch)

            # extract data from torch Variable, move to cpu, convert to numpy arrays
            # output_batch = output_batch.data.cpu().numpy()
            # labels_batch = labels_batch.data.cpu().numpy()

            # update the average loss
            metrics['loss'].update(loss.item())

            # compute all metrics on this batch
            self.compute_metrics(metrics, output_batch, labels_batch)

        metrics['visualize'](make_grid(data_batch), make_grid(output_batch))

        # Summary of metrics in log
        metrics_string = "".join([str(metric) for metric in metrics.values()])
        logging.info("- Val metrics : \n" + metrics_string)
        return metrics

    @staticmethod
    def assign_labels_to_clusters(kmeans, y_pred, y_true):
        """
        Assign class label to each K-means cluster using labeled data.
        The class label is based on the class of majority samples within a cluster.
        Unassigned clusters are labeled as -1.
        """
        print("Assigning labels to clusters ...", end=' ')
        start_time = time()

        labelled_clusters = []
        for i in range(kmeans.n_clusters):
            idx = np.where(y_pred == i)[0]
            if len(idx) != 0:
                labels_freq = np.bincount(y_true[idx])
                labelled_clusters.append(np.argmax(labels_freq))
            else:
                labelled_clusters.append(-1)
        print("Done in {:.2f} sec".format(time() - start_time))

        return np.asarray(labelled_clusters)

    @staticmethod
    def compute_kmeans_metrics(y_true, y_pred):
        """
        Compute kmeans metrics.
        """
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        return accuracy, f1

    def train_kmeans(self, train_dataloader, n_clusters, seed):
        """
        Train K-means model.
        """
        print("Training k-means ...", end=' ')
        kmeans = MiniBatchKMeans(init="k-means++", n_clusters=n_clusters, n_init=5, max_iter=100, random_state=seed)
        start_time = time.time()
        for i, (train_batch, label_batch) in enumerate(train_dataloader):
            train_batch = train_batch.to(self.device)
            train_embeddings = self.model.encoder(train_batch)
            train_embeddings = train_embeddings.detach().cpu().numpy()
            train_embeddings = np.squeeze(train_embeddings, -1)
            train_embeddings = np.squeeze(train_embeddings, -1)
            kmeans = kmeans.partial_fit(train_embeddings)

        print("Done in {:.2f} sec |".format(time.time() - start_time), end=' ')
        # print("Nb iter = {}, model inertia = {:.2f}".format(kmeans.n_iter_, kmeans.inertia_))

        return kmeans

    def eval_kmeans(self, kmeans, val_dataloader):
        """
        Predict labels and compare to true labels to compute the accuracy.
        """
        print("Evaluating k-means model ...", end=' ')
        start_time = time.time()
        y_preds = []
        for i, (data_batch, _) in enumerate(val_dataloader):
            data_batch = data_batch.to(self.device)
            val_embeddings = self.model.encoder(data_batch)
            val_embeddings = val_embeddings.detach().cpu().numpy()
            val_embeddings = np.squeeze(val_embeddings, -1)
            val_embeddings = np.squeeze(val_embeddings, -1)
            batch_preds = kmeans.predict(val_embeddings)
            y_preds.append(batch_preds)

        # accuracy, f1 = self.compute_kmeans_metrics(y_true, y_pred)
        # print("Done in {:.2f} sec | Accuracy: {:.2f} - F1: {:.2f}".format(time() - start_time, accuracy * 100, f1 * 100))

        return y_preds

#!/usr/bin/env python3
import os
import shutil
import logging
from tqdm import tqdm

from utils.misc import set_logger

import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from sklearn.cluster import MiniBatchKMeans, KMeans
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
        self.cluster_writer = SummaryWriter(os.path.join(self.experiment_path, 'tensorboard', 'cluster'))

    def compute_loss(self, output_batch, labels_batch):
        return self.loss(output_batch, labels_batch)

    @staticmethod
    def compute_metrics(metrics, param1, param2):
        for key, metric in metrics.items():
            if key == 'inertia':
                pass
            else:
                metric(param1, param2)

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

            # k-means stuff
            kmeans = self.train_kmeans(train_dataloader, k, seed)
            cluster_preds = self.predict_kmeans(kmeans, val_dataloader)
            _, cluster_metrics = self.eval_kmeans(kmeans,
                                                  cluster_preds,
                                                  val_dataloader.dataset.targets,
                                                  metrics.get('cluster'))

            cluster_metrics['inertia'](kmeans)

            #  Register cluster metrics to tensorboard
            self.register_metrics(cluster_metrics, epoch + 1)
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

        metrics['visualize'](make_grid(output_batch), make_grid(data_batch))

        # Summary of metrics in log
        metrics_string = "".join([str(metric) for metric in metrics.values()])
        logging.info("- Val metrics : \n" + metrics_string)
        return metrics

    def eval_kmeans(self, kmeans, cluster_pred, y_true, metrics):
        """
        Assign class label to each K-means cluster using labeled data.
        The class label is based on the class of majority samples within a cluster.
        Unassigned clusters are labeled as -1.
        After assigning computes kmeans accuracy and f1 score.
        """
        print("Assigning labels to clusters ...", end=' ')

        self.reset_metrics(metrics)

        start_time = time.time()

        labelled_clusters = []

        for i in range(0, kmeans.n_clusters):
            idx = np.where(cluster_pred == i)[0]
            if len(idx) != 0:
                labels_freq = np.bincount(y_true[idx])
                labelled_clusters.append(np.argmax(labels_freq))
            else:
                labelled_clusters.append(-1)
        labelled_clusters = np.asarray(labelled_clusters)
        y_preds = labelled_clusters[cluster_pred]

        self.compute_metrics(metrics, y_true.reshape(-1, 1), y_preds.reshape(-1, 1))
        accuracy, f1 = self.compute_kmeans_metrics(y_true.reshape(-1, 1), y_preds.reshape(-1, 1))
        print("Done in {:.2f} sec | Accuracy: {:.2f} - F1: {:.2f}".format(time.time() - start_time, accuracy * 100,
                                                                          f1 * 100))

        return labelled_clusters, metrics

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

        # kmeans = MiniBatchKMeans(init="k-means++", n_clusters=n_clusters, n_init=3, max_iter=100, random_state=seed)
        kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=5, max_iter=1000, random_state=seed, n_jobs=-1)
        start_time = time.time()
        train_embeddings = []

        for i, (train_batch, label_batch) in enumerate(train_dataloader):
            train_batch = train_batch.to(self.device)
            train_embeddings_batch = self.model.encoder(train_batch)
            train_embeddings_batch = train_embeddings_batch.detach().cpu().numpy()
            train_embeddings_batch = np.squeeze(train_embeddings_batch, -1)
            train_embeddings_batch = np.squeeze(train_embeddings_batch, -1)
            train_embeddings.append(train_embeddings_batch)

        train_embeddings = np.vstack(train_embeddings)
        kmeans = kmeans.fit(train_embeddings)

        print("Done in {:.2f} sec |".format(time.time() - start_time), end=' ')
        print("model inertia = {:.2f}".format(kmeans.inertia_))

        return kmeans

    def predict_kmeans(self, kmeans, val_dataloader):
        """
        Predict labels.
        """
        print("Evaluating k-means model ...", end=' ')

        y_preds = []

        for i, (data_batch, _) in enumerate(val_dataloader):
            data_batch = data_batch.to(self.device)
            val_embeddings = self.model.encoder(data_batch)
            val_embeddings = val_embeddings.detach().cpu().numpy()
            val_embeddings = np.squeeze(val_embeddings, -1)
            val_embeddings = np.squeeze(val_embeddings, -1)
            batch_preds = kmeans.predict(val_embeddings)
            y_preds.append(batch_preds)

        y_preds = np.squeeze(np.stack(y_preds), 0)

        return y_preds

#!/usr/bin/env python3
import os
import shutil
import logging
from tqdm import tqdm
from joblib import dump, load
from utils.misc import set_logger

import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter

from sklearn.cluster import  KMeans
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import GroupShuffleSplit
import warnings  # To mute scikit-learn warnings about f1 score.

warnings.filterwarnings("ignore")

import pdb
import time
import numpy as np


def match_labels(a, b):
    """
    Returns mappins from a to b that minimizes the distance between the input
    label vectors. Inputs must be the same length. The unique values from a + b
    are appended to a + b to ensure that both vectors contain all unique values.
    For details see section 2 of Lange T et al. 2004.
    E.g,
    a = [1,1,2,3,3,4,4,4,2]
    b = [2,2,3,1,1,4,4,4,3]
    optimal: 1 -> 2; 2 -> 3; 3 -> 1; 4 -> 4
    returns:
        1 2
        2 3
        3 1
        4 4
    Inspired by http://things-about-r.tumblr.com/post/36087795708/matching-clustering-solutions-using-the-hungarian
    """
    if len(a) != len(b):
        raise ValueError('length of a & b must be equal')

    ids_a = np.unique(a)
    ids_b = np.unique(b)

    # in some cases, a and b do not have the same number of unique entries. This
    # can happen if one of the two are predicted labels, and the data they were
    # predicted from had some very small classes which constitute outliers, and
    # are not predicted in the held out data. D should still contain an entry
    # for this unlikely class, and our mapping should account for it. To
    # facilitate this, we append all unique values from both a and b to each, so
    # a and b are garunteed to have at least one entry from all unique values
    n = max(len(ids_a), len(ids_b))  # may not be equal
    D = np.zeros((n, n))  # distance matrix
    a = np.hstack((np.hstack((a, ids_a)), ids_b))  # ensures no missing values
    b = np.hstack((np.hstack((b, ids_a)), ids_b))  #

    # constructs the distance matrix between a and b with appended values
    for x in np.arange(n):
        for y in np.arange(n):
            idx_a = np.where(a == x)[0]
            idx_b = np.where(b == y)[0]
            n_int = len(np.intersect1d(idx_a, idx_b))
            # distance = (# in cluster) - 2*sum(# in intersection)
            D[x, y] = (len(idx_a) + len(idx_b) - 2 * n_int)

    # permute labels w/ minimum weighted bipartite matching (hungarian method)
    idx_D_x, idx_D_y = linear_sum_assignment(D)
    mappings = np.hstack((np.atleast_2d(idx_D_x).T, np.atleast_2d(idx_D_y).T))

    return mappings


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

    def save_checkpoint(self, epoch, is_best, kmeans):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
        }
        state_filepath = os.path.join(self.experiment_path, 'models', 'last.pth.tar')
        model_filepath = os.path.join(self.experiment_path, 'models', 'last_model.pth')
        kmeans_filepath = os.path.join(self.experiment_path, 'models', 'last_kmeans.joblib')

        torch.save(state, state_filepath)
        torch.save(self.model, model_filepath)
        dump(kmeans, kmeans_filepath)
        if is_best:
            shutil.copyfile(state_filepath, os.path.join(self.experiment_path, 'models', 'best.pth.tar'))
            shutil.copyfile(model_filepath, os.path.join(self.experiment_path, 'models', 'best_model.pth'))
            shutil.copyfile(kmeans_filepath, os.path.join(self.experiment_path, 'models', 'best_kmeans.joblib'))

    @staticmethod
    def assign_labels_to_clusters(k, assignment_preds, eval_preds, assignment_ys):
        """
        Assign class label to each K-means cluster using labeled data.
        The class label is based on the class of majority samples within a cluster or hungarian method
        Unassigned clusters are labeled as -1.

        """
        print("Assigning labels to clusters ...", end=' ')

        labelled_clusters = []

        for i in range(0, k):
            idx = np.where(assignment_preds == i)[0]
            if len(idx) != 0:
                labels_freq = np.bincount(assignment_ys[idx])
                labelled_clusters.append(np.argmax(labels_freq))
            else:
                labelled_clusters.append(-1)

        labelled_clusters = np.asarray(labelled_clusters)
        y_preds = labelled_clusters[eval_preds]

        # hungarian method
        # mapping = match_labels(assignment_preds, assignment_ys)
        # yp_out = np.zeros(len(eval_preds), dtype=np.int32)
        # for c in np.arange(k):
        #    idx_map = np.where(eval_preds == c)
        #    yp_out[idx_map] = mapping[c, 1]

        return y_preds  # , yp_out

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
        best_val_acc = 0

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
            kmeans, cluster_preds = self.train_and_predict_kmeans(train_dataloader, val_dataloader, k, seed)

            regions = val_dataloader.dataset.regions
            ggg = GroupShuffleSplit(n_splits=1, test_size=0.5)

            for train_idx, test_idx in ggg.split(cluster_preds, val_dataloader.dataset.targets, regions):
                assignment_preds, eval_preds = cluster_preds[train_idx], cluster_preds[test_idx]
                assignment_ys, eval_ys = val_dataloader.dataset.targets[train_idx], \
                                         val_dataloader.dataset.targets[test_idx]
                assignment_regions, eval_regions = regions[train_idx], regions[test_idx]
            y_preds_freq = self.assign_labels_to_clusters(k, assignment_preds, eval_preds, assignment_ys)

            cluster_metrics = self.eval_kmeans(kmeans,
                                               eval_ys,
                                               y_preds_freq,
                                               metrics.get('cluster'))

            #  Register cluster metrics to tensorboard
            self.register_metrics(cluster_metrics, epoch + 1)

            # save best model
            val_acc = cluster_metrics['acc'].get_accuracy_cluster()
            is_best = val_acc >= best_val_acc
            if is_best:
                best_val_acc = val_acc
            self.save_checkpoint(epoch, is_best, kmeans)

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

            # update the average loss
            metrics['loss'].update(loss.item())

            # compute all metrics on this batch
            self.compute_metrics(metrics, output_batch, labels_batch)

        metrics['visualize'](make_grid(output_batch), make_grid(data_batch))

        # Summary of metrics in log
        metrics_string = "".join([str(metric) for metric in metrics.values()])
        logging.info("- Val metrics : \n" + metrics_string)
        return metrics

    @staticmethod
    def compute_kmeans_metrics(y_true, y_pred):
        """
        Compute kmeans metrics.
        """
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        return accuracy, f1

    def train_and_predict_kmeans(self, train_dataloader, val_dataloader, n_clusters, seed):
        """
        Train K-means model.
        """
        print("Training k-means ...", end=' ')

        kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=3, max_iter=1000, n_jobs=-1)
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

        print("Evaluating k-means model ...", end=' ')

        cluster_preds = []

        for i, (data_batch, _) in enumerate(val_dataloader):
            data_batch = data_batch.to(self.device)
            val_embeddings = self.model.encoder(data_batch)
            val_embeddings = val_embeddings.detach().cpu().numpy()
            val_embeddings = np.squeeze(val_embeddings, -1)
            val_embeddings = np.squeeze(val_embeddings, -1)
            batch_preds = kmeans.predict(val_embeddings)
            cluster_preds.append(batch_preds)

        cluster_preds = np.squeeze(np.stack(cluster_preds), 0)

        return kmeans, cluster_preds

    def eval_kmeans(self, kmeans, y_true, y_preds_freq, cluster_metrics):

        self.reset_metrics(cluster_metrics)
        cluster_metrics['inertia'](kmeans)

        self.compute_metrics(cluster_metrics, y_true.reshape(-1, 1), y_preds_freq.reshape(-1, 1))

        accuracy, f1 = self.compute_kmeans_metrics(y_true.reshape(-1, 1), y_preds_freq.reshape(-1, 1))
        print("Matching using max frequency:")
        print("Accuracy: {:.2f} - F1: {:.2f}".format(accuracy * 100,
                                                     f1 * 100))
        # print(80 * "-")
        # print("Matching using hungarian method:")
        # accuracy, f1 = self.compute_kmeans_metrics(y_true.reshape(-1, 1), y_preds_hung.reshape(-1, 1))
        # print("Accuracy: {:.2f} - F1: {:.2f}".format(accuracy * 100,
        #                                             f1 * 100))
        print(80 * "-")

        return cluster_metrics

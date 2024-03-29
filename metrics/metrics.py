import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


class Metric(object):
    """Base class for metric implementation"""
    def __init__(self, writer):
        self.writer = writer

    def reset(self):
        """Resets all attributes of the class to 0"""
        for var in vars(self):
            if var is not 'writer':
                setattr(self, var, 0)

    def dump(self):
        """Dump class attributes on a dictionary for JSON compatibility when saving
        Returns
        -------
        dict : dict
            Dictionary with attributes as keys and information as value.
        """
        return {var: str(getattr(self, var)) for var in vars(self)}

    def write_to_tensorboard(self, epoch):
        pass

    def is_best(self):
        raise NotImplementedError


class LossAverage(Metric):
    """Running average over loss value"""
    def __init__(self, writer):
        super(LossAverage, self).__init__(writer)
        self.steps = None
        self.total = None

    def __call__(self, output, labels):
        pass

    def __str__(self):
        """Return average loss when printing the class
        Returns
        -------
        str : str
            String containing the average loss for the current epoch.
        """
        return "Loss: {:0.3f}\n".format(self.total / self.steps)

    def get_average(self):
        return self.total / self.steps

    def update(self, val):
        """Update the loss average with the latest value
        Parameters
        ----------
        val : float
            Latest value of the loss after evaluating the models results.
        """
        self.steps += 1
        self.total += val

    def write_to_tensorboard(self, epoch):
        self.writer.add_scalar(self.__class__.__name__, self.get_average(), epoch)


class Accuracy(Metric):
    """Computes the accuracy for a given set of predictions"""
    def __init__(self, writer):
        super(Accuracy, self).__init__(writer)
        self.correct = None
        self.total = None

    def __call__(self, outputs, labels, params):
        """Updates the number of correct and total samples given the outputs of the model and the correct labels.
        Note:
        Adding the softmax function is not necessary to calculate the accuracy.
        Parameters
        ----------
        outputs : ndarray
            Model predictions.
        labels : ndarray
            Correct outputs.
        params : object
            Parameter class with general experiment information.
        """
        outputs = np.argmax(outputs, axis=1)
        self.total += float(labels.size)
        self.correct += np.sum(outputs == labels)

    def __str__(self):
        """Return sample accuracy when printing the class
        Returns
        -------
        str : str
            String containing the sample average accuracy for the summary steps.
        """
        return "Sample accuracy: {:0.3f} -- ({:.0f}/{:.0f})\n".format(self.correct/self.total, self.correct, self.total)

    def get_accuracy(self):
        """Returns average accuracy.
        Returns
        -------
        float : float
            Current accuracy for the sampled data.
        """
        return self.correct / self.total

    def write_to_tensorboard(self, epoch):
        self.writer.add_scalar(self.__class__.__name__, self.get_accuracy(), epoch)


class Visualize(Metric):
    """Returns images of data points and embeddings.
    """
    def __init__(self, writer):
        super(Visualize, self).__init__(writer)

    def __call__(self, network_images, labels):
        self.images = network_images
        self.labels = labels

    def write_to_tensorboard(self, epoch):
        self.writer.add_image(self.__class__.__name__ + '/output', self.images, epoch)
        self.writer.add_image(self.__class__.__name__ + '/labels', self.labels, epoch)


class ACC(Metric):
    """Returns clustering accuracy.
    """
    def __init__(self, writer):
        super(ACC, self).__init__(writer)

    def __call__(self, y_true, y_pred):
        """

        :param y_true: 1d nparray
            true labels.
        :param y_pred: 1d nparray
            cluster predictions.
        :return: float
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def get_accuracy_cluster(self):
        """Returns clustering accuracy.
        -------
            outputs : ndarray
                Model predictions.
            labels : ndarray
                Correct outputs.
            params : object
        """
        return accuracy_score(self.y_true, self.y_pred)

    def write_to_tensorboard(self, epoch):
        self.writer.add_scalar(self.__class__.__name__, self.get_accuracy_cluster(), epoch)


class F1score(Metric):
    """Returns f1 score of clustering module.
    """
    def __init__(self, writer):
        super(F1score, self).__init__(writer)

    def __call__(self, y_true, y_pred):
        """
        :param y_true: 1d nparray
            true labels.
        :param y_pred: 1d nparray
            cluster predictions.
        :return: float
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def get_f1(self):
        return f1_score(self.y_true, self.y_pred, average="weighted")

    def write_to_tensorboard(self, epoch):
        self.writer.add_scalar(self.__class__.__name__, self.get_f1(), epoch)


class Inertia(Metric):
    """Returns the sum of squared distances of samples
    to their closest cluster center.
    Returns
    -------
    float : float
    """
    def __init__(self, writer):
        super(Inertia, self).__init__(writer)

    def __call__(self, kmeans):
        self.inertia = kmeans.inertia_

    def write_to_tensorboard(self, epoch):
        self.writer.add_scalar(self.__class__.__name__, self.inertia, epoch)

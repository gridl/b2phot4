import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from joblib import load  # You can use Pickle or the serialization technique of your choice
sys.path.append("../")
from processing.loader import HoromaDataset
from models.cae import CAE

import pdb


def eval_model(model_path, dataset_dir, split):
    '''
    # MODIFY HERE #
    This function is meant to be an example

    '''

    # # SETUP MODEL # #
    # Load your best model
    print("\nLoading model from ({}).".format(model_path))
    model_state = os.path.join(model_path, 'best.pth.tar')
    network = CAE()

    checkpoint = torch.load(model_state)
    network.load_state_dict(checkpoint['model_state'])
    cluster = load(os.path.join(model_path, 'best_kmeans.joblib'))
    mapping = np.load(os.path.join(model_path, 'best_mapping.npy'))
    # # SETUP DATASET # #
    # Load requested dataset
    """ IMPORTANT # of example per splits.
    "train" = 150700
    "train_overlapped" = 544027
    "valid" = 480
    "valid_overlapped" = 1380
    "test" = 483

    Files available the test folder:
        test_regions_id.txt
        test_x.dat
        test_y.txt
        train_overlapped_regions_id.txt
        train_overlapped_x.dat
        train_overlapped_y.txt
        train_regions_id.txt
        train_x.dat
        train_y.txt
        valid_overlapped_regions_id.txt
        valid_overlapped_x.dat
        valid_overlapped_y.txt
        valid_regions_id.txt
        valid_x.dat
        valid_y.txt

    You need to load the right one according to the `split`.
    """
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = HoromaDataset(data_dir=dataset_dir, split=split, transform=transform, flattened=False)
    # Apply preprocessing if used

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # # INFERENCE # #
    # Use model on dataset to predict the class
    embeddings = []
    network.eval()

    for i, (batch, _) in enumerate(dataloader):
        batch_embeddings = network.encoder(batch)
        batch_embeddings = batch_embeddings.detach().cpu().numpy()
        embeddings.append(batch_embeddings)

    embeddings = np.vstack(embeddings)
    cluster_pred = cluster.predict(embeddings)
    target_pred = mapping[cluster_pred]
    # acc, f1 = accuracy_score(labels, target_pred), f1_score(labels, target_pred, average='weighted')

    pred = dataset.map_labels[target_pred]

    # # PREDICTIONS # #
    # Return the predicted classes as a numpy array of shape (nb_exemple, 1)
    """ Example:
    [['ES']
     ['EN']
     ['ES']]
    """
    return pred


if __name__ == "__main__":

    # Put your group name here
    group_name = "b2phot4"

    # model_path = "/rap/jvb-000-aa/COURS2019/etudiants/submissions/b2phot4/model/"
    model_path = "/home/user14/block2/b2phot4/configuration/kmeans_regular60/models/"
    # model_path should be the absolute path on shared disk to your best model.
    # You need to ensure that they are available to evaluators on Helios.

    #########################
    # DO NOT MODIFY - BEGIN #
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_dir", type=str, default="/rap/jvb-000-aa/COURS2019/etudiants/data/horoma/", help="Absolute path to the dataset directory.")
    parser.add_argument("-s", "--dataset_split", type=str, choices=['valid', 'test', 'train'], default="valid", help="Which split of the dataset should be loaded from `dataset_dir`.")
    parser.add_argument("-r", "--results_dir", type=str, default="./", help="Absolute path to where the predictions will be saved.")
    args = parser.parse_args()

    # Arguments validation
    if group_name is "b1phut_N":
        print("'group_name' is not set.\nExiting ...")
        exit(1)

    if model_path is None or not os.path.exists(model_path):
        print("'model_path' ({}) does not exists or unreachable.\nExiting ...".format(model_path))
        exit(1)

    if args.dataset_dir is None or not os.path.exists(args.dataset_dir):
        print("'dataset_dir' does not exists or unreachable..\nExiting ...")
        exit(1)

    y_pred = eval_model(model_path, args.dataset_dir, args.dataset_split)

    assert type(y_pred) is np.ndarray, "Return a numpy array"
    assert len(y_pred.shape) == 1, "Make sure ndim=1 for y_pred"

    results_fname = os.path.join(args.results_dir, "{}_pred_{}.txt".format(group_name, args.dataset_split))

    print('\nSaving results to ({})'.format(results_fname))
    np.savetxt(results_fname, y_pred, fmt='%s')
    # DO NOT MODIFY - END #
    #######################

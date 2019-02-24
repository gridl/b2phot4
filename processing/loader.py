import torch
from torch.utils.data import DataLoader, Dataset
import os.path as osp
import numpy as np
import pdb


class HoromaDataset(Dataset):

    def __init__(self, split, split_size, use_overlap=False):

        self.path_to_data = '/rap/jvb-000-aa/COURS2019/etudiants/data/horoma/'

        self.path_to_split = osp.join(self.path_to_data + split + '_x.dat')
        self.path_to_labels = osp.join(self.path_to_data + split + '_y.txt')

        if use_overlap:
            self.path_to_split = osp.join(self.path_to_data + split + '_overlapped_x.dat')
            self.path_to_labels = osp.join(self.path_to_data + split + '_overlapped_y.txt')

        self.data = np.memmap(self.path_to_split, dtype='float32', mode='r', shape=(split_size, 32, 32, 3))
        self.data = torch.from_numpy(self.data)
        self.data = self.data.permute([0, -1, 1, 2])
        self.str_labels = np.genfromtxt(self.path_to_labels, dtype='str')
        self.lookup, self.id_labels = np.unique(self.str_labels, return_inverse=True)
        self.id_labels = torch.from_numpy(self.id_labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        sample = self.data[item, :, :, :]
        label = self.id_labels[item]

        return sample, label


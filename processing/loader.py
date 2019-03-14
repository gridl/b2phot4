import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os.path as osp
import numpy as np
import pdb


class HoromaDataset(Dataset):

    def __init__(self, data_dir='/rap/jvb-000-aa/COURS2019/etudiants/data/horoma/',
                 split="train", subset=None, skip=0, flattened=True, transform=None):
        """
        Args:
            data_dir: Path to the directory containing the samples.
            split: Which split to use. [train, valid, test]
            subset: How many elements will be used. Default: all.
            skip: How many element to skip before taking the subset.
            flattened: If True return the images in a flatten format.
        """
        nb_channels = 3
        height = 32
        width = 32
        datatype = "uint8"

        if split == "train":
            self.nb_exemples = 150900
        elif split == "valid":
            self.nb_exemples = 480
        elif split == "test":
            self.nb_exemples = 498
        elif split == "train_overlapped":
            self.nb_exemples = 544749
        elif split == "valid_overlapped":
            self.nb_exemples = 1331
        else:
            raise("Dataset: Invalid split. Must be [train, valid, test, train_overlapped, valid_overlapped]")

        filename_x = osp.join(data_dir, "{}_x.dat".format(split))
        filename_y = osp.join(data_dir, "{}_y.txt".format(split))
        filename_regions = osp.join(data_dir, "{}_regions_id.txt").format(split)
        self.targets = None
        if osp.exists(filename_y) and not split.startswith("train"):
            pre_targets = np.loadtxt(filename_y, 'U2')

            if subset is None:
                pre_targets = pre_targets[skip: None]
            else:
                pre_targets = pre_targets[skip: skip + subset]

            self.map_labels = np.unique(pre_targets)
            self.targets = np.asarray([np.where(self.map_labels == t)[0][0] for t in pre_targets])

        if osp.exists(filename_regions):
            self.regions = np.genfromtxt(filename_regions, dtype=np.int32)
        self.data = np.memmap(filename_x,
                              dtype=datatype,
                              mode="r",
                              shape=(self.nb_exemples, height, width, nb_channels))
        self.transform = transform

        if subset is None:
            self.data = self.data[skip: None]
        else:
            self.data = self.data[skip: skip + subset]

        if flattened:
            self.data = self.data.reshape(len(self.data), -1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # permute to get in pytorch format
        if self.targets is not None:
            return self.transform(self.data[index]), torch.Tensor([self.targets[index]])
        else:
            return self.transform(self.data[index]),\
                   self.transform(self.data[index]),


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    seed = 4242
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    transforms = transforms.Compose([transforms.ToTensor()])
    dataset_folder = "/rap/jvb-000-aa/COURS2019/etudiants/data/horoma/"
    subset = None  # 222
    skip = 0
    batch_size = 100

    for split in ["valid", "valid_overlapped", "train", "train_overlapped"]:
        dataset = HoromaDataset(dataset_folder, split, subset=subset, skip=skip, flattened=False, transform=transforms)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        print("\n{dataset} len : {} | {dataset}loader len : {}".format(len(dataset), len(data_loader), dataset=split))
        print("Iterating Loader:")
        if split.startswith("train"):
            for i, x in enumerate(data_loader):
                print(i, x.size())
        else:
            for i, (x, y) in enumerate(data_loader):
                print(i, x.size(), y.size())
            print(y[0])

#        plt.imshow(x[0].type(torch.LongTensor))
#        plt.show()

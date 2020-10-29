from torch.utils.data import Dataset
from torchvision.transforms import transforms
from sklearn.model_selection import train_test_split

import os
import glob
import torch
import numpy as np
from PIL import Image
import pdb


class LocalDataset(Dataset):
    def __init__(
        self,
        root: str,
        dataset_name: str,
        target_transform,
        train=True,
        random_state=None,
        split=True,
        random_effect=True,
    ):
        super(Dataset, self).__init__()
        self.target_transform = target_transform

        self.classes = [0, 1]
        self.root = root
        self.train = train  # training set or test set
        # self.dataset_path = os.path.join(self.root, self.dataset_name)
        # class_idx/image
        X = np.array(glob.glob(os.path.join(self.root, "*/*.[jp][pn][g]")))
        y = [int(i.split("/")[-2]) for i in X]
        y = np.array(y)
        if split:
            idx_norm = y == 0
            idx_out = y != 0

            # 80% data for training and 20% for testing; keep outlier ratio
            # pdb.set_trace()
            X_train_norm, X_test_norm, y_train_norm, y_test_norm = train_test_split(
                X[idx_norm], y[idx_norm], test_size=0.1, random_state=random_state, stratify=y[idx_norm]
            )
            X_train_out, X_test_out, y_train_out, y_test_out = train_test_split(
                X[idx_out], y[idx_out], test_size=0.1, random_state=random_state, stratify=y[idx_out]
            )
            X_train = np.concatenate((X_train_norm, X_train_out))
            X_test = np.concatenate((X_test_norm, X_test_out))
            y_train = np.concatenate((y_train_norm, y_train_out))
            y_test = np.concatenate((y_test_norm, y_test_out))

            if self.train:
                self.data = X_train
                self.targets = torch.tensor(y_train, dtype=torch.int64)
            else:
                self.data = X_test
                self.targets = torch.tensor(y_test, dtype=torch.int64)
        else:
            self.data = X
            self.targets = torch.tensor(y, dtype=torch.int64)

        self.semi_targets = torch.zeros_like(self.targets)
        # for training we will add brightness variance
        if random_effect:
            self.transform = transforms.Compose(
                [
                    # transforms.ColorJitter(
                    #     brightness=0.5 + int(np.random.rand(1)), contrast=0.5 + int(np.random.rand(1))
                    # ),
                    # saturation=0.5 + int(np.random.rand(1)),
                    # hue=0.5 + int(np.random.rand(1))),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                ]
            )
        # for testing
        else:
            self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """
        data = Image.open(self.data[index])
        data = self.transform(data)
        sample, target, semi_target = data, 0 if self.targets[index] == 0 else 1, int(self.semi_targets[index])
        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)

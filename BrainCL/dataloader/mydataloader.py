# coding=utf-8


import h5py
import numpy as np
import torch
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    """Custom dataset for FCN samples and labels."""
    def __init__(self, dataset_path, indices):
        super(MyDataset, self).__init__()
        self.indices = indices
        with h5py.File(dataset_path, 'r') as f:
            self.fcns = torch.from_numpy(f['fcn'][self.indices]).float()
            self.labels = torch.tensor(f['label'][self.indices]).float()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        fcn = self.fcns[idx]
        label = self.labels[idx]
        return fcn, label


class MyDataLoader(object):
    """Build cross-validation loaders."""
    def __init__(self, dataset_path, n_repeats=10, n_splits=10, seed=42, batch_size=16):
        self.dataset_path = dataset_path
        self._n_repeats = n_repeats
        self._n_splits = n_splits
        self.seed = seed
        self.batch_size = batch_size
        self.joint_label = self._load_joint_labels()
        self.rskf = RepeatedStratifiedKFold(n_repeats=self._n_repeats, n_splits=self._n_splits, random_state=self.seed)
        self.folds = list(self.rskf.split(np.zeros(len(self.joint_label)), self.joint_label))

    def _load_joint_labels(self):
        with h5py.File(self.dataset_path, 'r') as f:
            labels = f['label'][:]
            sites = f['site'][:]
        labels = np.array([f'{round(label)}' for label in labels])
        sites = np.array([site.decode(encoding='utf-8', errors='strict') for site in sites])
        joint_labels = np.array([f'{label}_{site}' for label, site in zip(labels, sites)])
        return joint_labels

    @property
    def n_repeats(self):
        return self._n_repeats

    @property
    def n_splits(self):
        return self._n_splits

    def get_cur_dataloader(self, repeat_idx, fold_idx):
        """Return the training, validation, and test loaders for the current fold."""
        assert 0 <= repeat_idx < self._n_repeats
        assert 0 <= fold_idx < self._n_splits

        global_fold_idx = repeat_idx * self._n_splits + fold_idx
        train_idx, test_idx = self.folds[global_fold_idx]
        train_idx, val_idx = train_test_split(
            train_idx,
            test_size=0.15,
            stratify=self.joint_label[train_idx],
            random_state=self.seed + global_fold_idx,
        )
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)

        train_dataset = MyDataset(self.dataset_path, train_idx)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                                      pin_memory=True)

        val_dataset = MyDataset(self.dataset_path, val_idx)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4,
                                    pin_memory=True)

        test_dataset = MyDataset(self.dataset_path, test_idx)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4,
                                     pin_memory=True)

        return train_dataloader, val_dataloader, test_dataloader

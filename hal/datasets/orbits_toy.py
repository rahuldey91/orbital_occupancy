# orbits_toy.py

import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader

import os
import torch
from random import shuffle
from scipy.io import loadmat
import numpy as np

__all__ = ['Orbits']

def prepare_data(root=None):
    circle_points = loadmat(os.path.join(root, 'pointsCircumferenceCircle.mat'))['surfaceCircle']
    ellipse_points = loadmat(os.path.join(root, 'pointsCircumferenceEllipse.mat'))['surfaceEllipse']
    circle_band = loadmat(os.path.join(root, 'pointsInsideCircleBand.mat'))['Region1']
    ellipse_band = loadmat(os.path.join(root, 'pointsInsideEllipseBand.mat'))['Region2']
    circle_outside = loadmat(os.path.join(root, 'pointsOutsideCircleBand.mat'))['Region34']
    ellipse_outside = loadmat(os.path.join(root, 'pointsOutsideEllipseBand.mat'))['Region56']

    circle_inside = np.concatenate((circle_points, circle_band), 0)
    circle_inside = np.concatenate((circle_inside, np.ones((circle_inside.shape[0], 1))), 1)
    circle_outside = np.concatenate((circle_outside, np.zeros((circle_outside.shape[0], 1))), 1)
    circle_all = np.concatenate((circle_inside, circle_outside), 0)

    ellipse_inside = np.concatenate((ellipse_points, ellipse_band), 0)
    ellipse_inside = np.concatenate((ellipse_inside, np.ones((ellipse_inside.shape[0], 1))), 1)
    ellipse_outside = np.concatenate((ellipse_outside, np.zeros((ellipse_outside.shape[0], 1))), 1)
    ellipse_all = np.concatenate((ellipse_inside, ellipse_outside), 0)

    np.random.shuffle(circle_all)
    np.random.shuffle(ellipse_all)

    return circle_all, ellipse_all

def split_dataset(data, split):
    length = data.shape[0]
    train_len = int(length * split)
    data_train = data[:train_len]
    data_val = data[train_len:]
    return data_train, data_val

class OrbitsDataset:
    def __init__(self, points1, points2):
        # super().__init__()
        self.points1 = points1
        self.points2 = points2

    def __len__(self):
        return self.points1.shape[0]

    def __getitem__(self, idx):
        return self.points1[idx], self.points2[idx]

class Orbits(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts
        self.data_fraction = opts.data_fraction
        circle, ellipse = prepare_data(root=self.opts.dataroot)
        length = int(circle.shape[0] * self.data_fraction)
        circle = torch.tensor(circle[:length]).float()
        ellipse = torch.tensor(ellipse[:length]).float()
        self.circle_train, self.circle_val = split_dataset(circle, split=self.opts.split_train)
        self.ellipse_train, self.ellipse_val = split_dataset(ellipse, split=self.opts.split_train)
        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

    def train_dataloader(self):
        dataset = OrbitsDataset(self.circle_train, self.ellipse_train)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_train,
            shuffle=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        dataset = OrbitsDataset(self.circle_val, self.ellipse_val)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        dataset = OrbitsDataset(self.circle_val, self.ellipse_val)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

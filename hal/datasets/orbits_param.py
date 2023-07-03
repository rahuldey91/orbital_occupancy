# orbits_param.py

import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader

import os
import torch
from random import shuffle
from scipy.io import loadmat
import numpy as np

__all__ = ['OrbitsParameterized']


class OrbitsDataset:
    def __init__(self, opts, datalen):
        self.datalen = datalen
        self.circle_r = opts.circle_r
        self.circle_o = np.array(opts.circle_o)
        self.ellipse1 = opts.ellipse1
        self.ellipse2 = opts.ellipse2
        self.ellipse_o = np.array(opts.ellipse_o)

        self.points_range = np.array(opts.points_range)
        points_c = np.random.uniform(self.points_range[0], self.points_range[1], size=(datalen, 2)).astype(np.float32)
        points_e = np.random.uniform(self.points_range[0], self.points_range[1], size=(datalen, 2)).astype(np.float32)
        self.points_c = np.concatenate((points_c, self.is_inside_circle(points_c)[:,None]), 1)
        self.points_e = np.concatenate((points_e, self.is_inside_ellipse(points_e)[:,None]), 1)

    def is_inside_circle(self, point):
        dist = np.linalg.norm(point - self.circle_o, 2, axis=1)
        return ((dist >= self.circle_r[0]) * (dist <= self.circle_r[1])).astype(np.float32)

    def is_inside_ellipse(self, point):
        point = point - self.ellipse_o
        outside_inner = (point[:,0] / self.ellipse1[0]) ** 2 + (point[:,1] / self.ellipse1[1]) ** 2 - 1 >= 0
        inside_outer = (point[:,0] / self.ellipse2[0]) ** 2 + (point[:,1] / self.ellipse2[1]) ** 2 - 1 <= 0
        return (outside_inner * inside_outer).astype(np.float32)

    def __len__(self):
        return self.datalen

    def __getitem__(self, idx):
        point = np.random.uniform()
        return self.points_c[idx], self.points_e[idx]


class OrbitsParameterized(pl.LightningDataModule):
    def __init__(self, opts):
        super().__init__()
        self.opts = opts

        if opts.ngpu == 0:
            self.pin_memory = False
        else:
            self.pin_memory = True

    def train_dataloader(self):
        dataset = OrbitsDataset(self.opts, self.opts.train_len)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_train,
            shuffle=True,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def val_dataloader(self):
        dataset = OrbitsDataset(self.opts, self.opts.val_len)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

    def test_dataloader(self):
        dataset = OrbitsDataset(self.opts, self.opts.val_len)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.opts.batch_size_test,
            shuffle=False,
            num_workers=self.opts.nthreads,
            pin_memory=self.pin_memory,
        )
        return loader

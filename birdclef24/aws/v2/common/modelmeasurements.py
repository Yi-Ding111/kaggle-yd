import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import random
from typing import List
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

import torchaudio

import torch

from torch.utils.data import DataLoader, TensorDataset

import lightning as L

import datasets

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from pathlib import Path
import multiprocessing
import colorednoise as cn
import torch.nn as nn
import librosa
from torch.distributions import Beta
from torch_audiomentations import Compose, PitchShift, Shift, OneOf, AddColoredNoise

import timm
from torchinfo import summary

import torch.nn.functional as F

from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    OneCycleLR,
)
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from lightning.pytorch.loggers import MLFlowLogger

from sklearn.metrics import roc_auc_score


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, sample_weight=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # category weights
        self.sample_weight = sample_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.weight)
        p_t = torch.exp(-ce_loss)  # Modulating Factor
        loss = (1 - p_t) ** self.gamma * ce_loss

        if self.sample_weight is not None:
            loss *= self.sample_weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


from sklearn.metrics import roc_auc_score
import torch
import numpy as np


def compute_roc_auc(preds, targets):
    preds = (
        torch.sigmoid(preds).detach().cpu().numpy()
    )  # Make sure preds is a probability value
    targets = targets.detach().cpu().numpy()

    # Make sure targets are binary labels
    targets = (targets >= 0.5).astype(int)

    auc_scores = []
    for i in range(targets.shape[1]):
        if (
            np.unique(targets[:, i]).size > 1
        ):  # Make sure there are at least two different categories
            auc = roc_auc_score(targets[:, i], preds[:, i])
            auc_scores.append(auc)
        else:
            # print(f"Warning: Only one class present in class {i}, AUC cannot be computed.")
            auc_scores.append(0.5)  # Optionally give a default value

    if len(auc_scores) > 0:
        average_auc = np.mean(auc_scores)
    else:
        average_auc = 0.0  # Returns 0 when no AUC can be calculated

    return average_auc

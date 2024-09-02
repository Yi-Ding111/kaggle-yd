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

from torch.utils.data import DataLoader,TensorDataset

import lightning as L

import datasets

from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler

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
from lightning.pytorch.callbacks  import ModelCheckpoint, EarlyStopping

from lightning.pytorch.loggers import MLFlowLogger

from sklearn.metrics import roc_auc_score


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, sample_weight=None,reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # categories weights
        self.sample_weight=sample_weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        p_t = torch.exp(-ce_loss) # Modulating Factor
        loss = (1 - p_t) ** self.gamma * ce_loss

        if self.sample_weight is not None:
            loss *= self.sample_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
        


def compute_roc_auc(preds, targets):
    preds = torch.sigmoid(preds)  # Assuming binary or multi-label classification
    preds = preds.detach().cpu().numpy()  # Detach and convert to numpy
    targets = targets.detach().cpu().numpy()
    
    # Compute ROC-AUC on a per-class basis and average
    auc_scores = []
    for i in range(targets.shape[1]):  # Loop through classes
        if targets[:, i].sum() > 0:  # Only score classes with positive labels
            auc = roc_auc_score(targets[:, i], preds[:, i])
            auc_scores.append(auc)
    
    if len(auc_scores) > 0:
        return sum(auc_scores) / len(auc_scores)  # Return macro average
    else:
        return 0.0  # Handle cases where no class has positives
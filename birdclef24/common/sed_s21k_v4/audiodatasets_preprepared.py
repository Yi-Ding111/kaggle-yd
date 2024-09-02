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

from common.sed_s21k_v4.audioprocess import rating_value_interplote, audio_weight
from common.sed_s21k_v4.audiotransform import CustomCompose,CustomOneOf,NoiseInjection,GaussianNoise,PinkNoise,AddGaussianNoise,AddGaussianSNR
from common.sed_s21k_v4.audiotransform import read_audio, Mixup, mel_transform,image_delta, Mixup2

from glob import glob

class BirdclefDataset(Dataset):
    def __init__(self, data_dir):
        self.clip_files = glob(os.path.join(data_dir, "clips_batch_*.pt"))
        self.label_files = glob(os.path.join(data_dir, "labels_batch_*.pt"))
        self.weight_files = glob(os.path.join(data_dir, "weights_batch_*.pt"))
        self.clip_files.sort()
        self.label_files.sort()
        self.weight_files.sort()

    def __len__(self):
        return len(self.clip_files)

    def __getitem__(self, index):
        clips = torch.load(self.clip_files[index])
        labels = torch.load(self.label_files[index])
        weights = torch.load(self.weight_files[index])
        return clips, labels, weights
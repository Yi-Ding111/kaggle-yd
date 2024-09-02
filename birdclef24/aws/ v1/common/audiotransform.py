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


device = "cuda"

# Need to interpolate missing values ​​for ratings in metadata csv files


def rating_value_interplote(df: pd.DataFrame):
    """
    interplote Nan values for rating col in metadata csv

    parameters:
        df: the df of the metadata csv file

    rating col means the quality of the corresponding audio file
        5 is high quality
        1 is low quality
        0 is without defined quality level
    """

    if df["rating"].isna().sum() > 0:  # represents having missing values
        df["rating"].fillna(0, inplace=True)

    # Randomly assign a value to all places where the value is 0, choosing from the specified choices
    mask = df["rating"] == 0  # create a bool mask, indicate which positions are 0

    choices = np.arange(
        0.5, 5.1, 0.5
    ).tolist()  # [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    random_values = np.random.choice(
        choices, size=mask.sum()
    )  # Generate random numbers for these 0 values
    df.loc[mask, "rating"] = (
        random_values  # Fill the generated random numbers back into the corresponding positions of the original DataFrame
    )

    return df


# Calculating the weight of each audio file by rating helps model training
def audio_weight(df):
    """
    calculate the weight corresponding to each audio file through the rating value

    Because each audio has different quality level, we use weight to affect the inportance of each audio in models,
    the lower the quality of the audio, the lower the weight
    """
    # Through rating, we calculate the credibility of each audio and express it through weight.
    # The purpose of this is to improve the model by increasing the weight of high-quality audio and reducing the weight of low-quality audio.
    df["audio_weight"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)

    return df


class Mixup(nn.Module):
    def __init__(self, mix_beta, mixup_prob, mixup_double):
        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixup_prob = mixup_prob
        self.mixup_double = mixup_double

    def forward(self, X, Y, weight=None):
        p = torch.rand((1,))[
            0
        ]  # Generate a random number p and compare it with mixup_prob to decide whether to mix.
        if p < self.mixup_prob:
            bs = X.shape[0]  # batch size
            n_dims = len(X.shape)
            perm = torch.randperm(bs)
            # generate a random permutation, for selecting samples from the current batch to mix randomly.

            p1 = torch.rand((1,))[
                0
            ]  # If the random number p1 (determines whether to perform double mixing) is less than mixup_double, perform a single mix. Otherwise, perform double mixing:
            if p1 < self.mixup_double:
                X = X + X[perm]
                Y = Y + Y[perm]
                Y = torch.clamp(
                    Y, 0, 1
                )  # Use torch.clamp to clamp the values ​​of Y between 0 and 1 (suitable for probabilistic or binary labels).

                if weight is None:
                    return X, Y
                else:
                    weight = 0.5 * weight + 0.5 * weight[perm]
                    return X, Y, weight
            else:
                perm2 = torch.randperm(bs)
                X = X + X[perm] + X[perm2]
                Y = Y + Y[perm] + Y[perm2]
                Y = torch.clamp(Y, 0, 1)

                if weight is None:
                    return X, Y
                else:
                    weight = (
                        1 / 3 * weight + 1 / 3 * weight[perm] + 1 / 3 * weight[perm2]
                    )
                    return X, Y, weight
        else:
            if weight is None:
                return X, Y
            else:
                return X, Y, weight


def mel_transform(
    sample_rate: float,
    audio: torch.Tensor,
    window_size: float = 0.04,
    hop_size: float = 0.02,
    n_mels: int = 40,
) -> torch.Tensor:
    """
    transform audio data into mel sepctrogram
    """
    # Determine window size and frame shift
    # window_size = 0.04 # 40 milliseconds
    # hop_size = 0.02 # 20 milliseconds, usually half the window size
    n_fft = int(
        window_size * sample_rate
    )  # Convert window size to number of sampling points
    hop_length = int(
        hop_size * sample_rate
    )  # Convert frame shift to sampling point number

    # Calculate Mel spectrum
    # n_mels = 40 # The number of Mel filters

    # Set up the Mel Spectrogram converter
    mel_transformer = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0,
        f_max=16000,
    ).to(device)

    melspec = mel_transformer(audio)

    return melspec.to(device)


def compute_deltas(
    specgram: torch.Tensor, win_length: int = 5, mode: str = "replicate"
) -> torch.Tensor:
    """Compute delta coefficients of a tensor, usually a spectrogram.

    Args:
        specgram (Tensor): Tensor of audio of dimension (..., freq, time)
        win_length (int, optional): The window length used for computing delta (Default: 5)
        mode (str, optional): Mode parameter passed to padding (Default: "replicate")

    Returns:
        Tensor: Tensor of deltas of dimension (..., freq, time)
    """
    device = specgram.device
    dtype = specgram.dtype

    # pack batch
    shape = specgram.size()
    specgram = specgram.reshape(1, -1, shape[-1])

    assert win_length >= 3
    n = (win_length - 1) // 2
    denom = n * (n + 1) * (2 * n + 1) / 3

    specgram = torch.nn.functional.pad(specgram, (n, n), mode=mode)

    # Create the kernel tensor, making sure it is on the same device as the input tensor
    kernel = torch.arange(-n, n + 1, 1, dtype=dtype, device=device).repeat(
        specgram.shape[1], 1, 1
    )

    output = (
        torch.nn.functional.conv1d(specgram, kernel, groups=specgram.shape[1]) / denom
    )

    # unpack batch
    output = output.reshape(shape)

    return output


def make_delta(input_tensor: torch.Tensor):
    input_tensor = input_tensor.transpose(3, 2)
    input_tensor = compute_deltas(input_tensor)
    input_tensor = input_tensor.transpose(3, 2)
    return input_tensor


def image_delta(x):
    delta_1 = make_delta(x)
    delta_2 = make_delta(delta_1)
    x = torch.cat([x, delta_1, delta_2], dim=1)
    return x


class Mixup2(nn.Module):
    def __init__(self, mix_beta, mixup2_prob):
        super(Mixup2, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)
        self.mixup2_prob = mixup2_prob

    def forward(self, X, Y, weight=None):
        p = torch.rand((1,))[0]
        if p < self.mixup2_prob:
            bs = X.shape[0]
            n_dims = len(X.shape)
            perm = torch.randperm(bs)
            coeffs = self.beta_distribution.rsample(torch.Size((bs,))).to(device)

            if n_dims == 2:
                X = coeffs.view(-1, 1) * X + (1 - coeffs.view(-1, 1)) * X[perm]
            elif n_dims == 3:
                X = coeffs.view(-1, 1, 1) * X + (1 - coeffs.view(-1, 1, 1)) * X[perm]
            else:
                X = (
                    coeffs.view(-1, 1, 1, 1) * X
                    + (1 - coeffs.view(-1, 1, 1, 1)) * X[perm]
                )
            Y = coeffs.view(-1, 1) * Y + (1 - coeffs.view(-1, 1)) * Y[perm]
            # Y = Y + Y[perm]
            # Y = torch.clamp(Y, 0, 1)

            if weight is None:
                return X, Y
            else:
                weight = coeffs.view(-1) * weight + (1 - coeffs.view(-1)) * weight[perm]
                return X, Y, weight
        else:
            if weight is None:
                return X, Y
            else:
                return X, Y, weight

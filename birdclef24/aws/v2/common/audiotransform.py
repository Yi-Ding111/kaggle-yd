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


def read_audio(path: str):
    """
    Read an OGG file using torchaudio and return the waveform tensor and sample rate.

    Parameters:
        path: Path to the .ogg file

    Returns:
        waveform: Tensor representing the waveform
        sample_rate: Sample rate of the audio file
    """
    audio, sample_rate = torchaudio.load(path)
    return audio, sample_rate


class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y: np.ndarray):
        raise NotImplementedError


class CustomCompose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray):
        for trns in self.transforms:
            y = trns(y)
        return y


class CustomOneOf:
    def __init__(self, transforms: list, p=1.0):
        self.transforms = transforms
        self.p = p

    def __call__(self, y: np.ndarray):
        if np.random.rand() < self.p:
            n_trns = len(self.transforms)
            trns_idx = np.random.choice(n_trns)
            trns = self.transforms[trns_idx]
            y = trns(y)
        return y


class GaussianNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=40.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y**2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise**2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoiseSNR(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5.0, max_snr=20.0, **kwargs):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y**2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise**2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


class VolumeControl(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, db_limit=10, mode="uniform"):
        super().__init__(always_apply, p)

        assert mode in [
            "uniform",
            "fade",
            "fade",
            "cosine",
            "sine",
        ], "`mode` must be one of 'uniform', 'fade', 'cosine', 'sine'"

        self.db_limit = db_limit
        self.mode = mode

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.db_limit, self.db_limit)
        if self.mode == "uniform":
            db_translated = 10 ** (db / 20)
        elif self.mode == "fade":
            lin = np.arange(len(y))[::-1] / (len(y) - 1)
            db_translated = 10 ** (db * lin / 20)
        elif self.mode == "cosine":
            cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * cosine / 20)
        else:
            sine = np.sin(np.arange(len(y)) / len(y) * np.pi * 2)
            db_translated = 10 ** (db * sine / 20)
        augmented = y * db_translated
        return augmented


class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5, sr=32000):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20, sr=32000):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y**2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise**2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20, sr=32000):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y**2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise**2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1, sr=32000):
        super().__init__(always_apply, p)
        self.max_rate = max_rate
        self.sr = sr

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate)
        return augmented


def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10 ** (db / 20)
    else:
        return 10 ** (db / 10)


def volume_down(y: np.ndarray, db: float):
    """
    Low level API for decreasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to decrease
    Returns
    -------
    applied: numpy.ndarray
        audio with decreased volume
    """
    applied = y * _db2float(-db)
    return applied


def volume_up(y: np.ndarray, db: float):
    """
    Low level API for increasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to increase
    Returns
    -------
    applied: numpy.ndarray
        audio with increased volume
    """
    applied = y * _db2float(db)
    return applied


class RandomVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        if db >= 0:
            return volume_up(y, db)
        else:
            return volume_down(y, db)


class CosineVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
        dbs = _db2float(cosine * db)
        return y * dbs


class AddGaussianNoise(AudioTransform):
    """Add gaussian noise to the samples"""

    supports_multichannel = True

    def __init__(
        self, always_apply=False, min_amplitude=0.001, max_amplitude=0.015, p=0.5
    ):
        """
        :param min_amplitude: Minimum noise amplification factor
        :param max_amplitude: Maximum noise amplification factor
        :param p:
        """
        super().__init__(always_apply, p)
        assert min_amplitude > 0.0
        assert max_amplitude > 0.0
        assert max_amplitude >= min_amplitude
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def apply(self, samples: np.ndarray, sample_rate=32000):
        amplitude = np.random.uniform(self.min_amplitude, self.max_amplitude)
        noise = np.random.randn(*samples.shape).astype(np.float32)
        samples = samples + amplitude * noise
        return samples


class AddGaussianSNR(AudioTransform):
    """
    Add gaussian noise to the input. A random Signal to Noise Ratio (SNR) will be picked
    uniformly in the decibel scale. This aligns with human hearing, which is more
    logarithmic than linear.
    """

    supports_multichannel = True

    def __init__(
        self,
        always_apply=False,
        min_snr_in_db: float = 5.0,
        max_snr_in_db: float = 40.0,
        p: float = 0.5,
    ):
        """
        :param min_snr_in_db: Minimum signal-to-noise ratio in dB. A lower number means more noise.
        :param max_snr_in_db: Maximum signal-to-noise ratio in dB. A greater number means less noise.
        :param p: The probability of applying this transform
        """
        super().__init__(always_apply, p)
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db

    def apply(self, samples: np.ndarray, sample_rate=32000):
        snr = np.random.uniform(self.min_snr_in_db, self.max_snr_in_db)

        clean_rms = np.sqrt(np.mean(np.square(samples)))

        a = float(snr) / 20
        noise_rms = clean_rms / (10**a)

        noise = np.random.normal(0.0, noise_rms, size=samples.shape).astype(np.float32)
        return samples + noise


class Normalize(AudioTransform):
    """
    Apply a constant amount of gain, so that highest signal level present in the sound becomes
    0 dBFS, i.e. the loudest level allowed if all samples must be between -1 and 1. Also known
    as peak normalization.
    """

    supports_multichannel = True

    def __init__(self, always_apply=False, apply_to: str = "all", p: float = 0.5):
        super().__init__(always_apply, p)
        assert apply_to in ("all", "only_too_loud_sounds")
        self.apply_to = apply_to

    def apply(self, samples: np.ndarray, sample_rate=32000):
        max_amplitude = np.amax(np.abs(samples))
        if self.apply_to == "only_too_loud_sounds" and max_amplitude < 1.0:
            return samples

        if max_amplitude > 0:
            return samples / max_amplitude
        else:
            return samples


class NormalizeMelSpec(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, X):
        mean = X.mean((1, 2), keepdim=True)
        std = X.std((1, 2), keepdim=True)
        Xstd = (X - mean) / (std + self.eps)
        norm_min, norm_max = Xstd.min(-1)[0].min(-1)[0], Xstd.max(-1)[0].max(-1)[0]
        fix_ind = (norm_max - norm_min) > self.eps * torch.ones_like(
            (norm_max - norm_min)
        )
        V = torch.zeros_like(Xstd)
        if fix_ind.sum():
            V_fix = Xstd[fix_ind]
            norm_max_fix = norm_max[fix_ind, None, None]
            norm_min_fix = norm_min[fix_ind, None, None]
            V_fix = torch.max(
                torch.min(V_fix, norm_max_fix),
                norm_min_fix,
            )
            # print(V_fix.shape, norm_min_fix.shape, norm_max_fix.shape)
            V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
            V[fix_ind] = V_fix
        return V


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
            # n_dims = len(X.shape)
            perm = torch.randperm(
                bs
            )  # Generate a random permutation for randomly selecting samples from the current batch for mixing.

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
    hop_size: float = 0.01,
    n_mels: int = 40,
) -> torch.Tensor:
    """
    transform audio data into mel sepctrogram
    """
    # Determine window size and frame shift
    # window_size = 0.04  # 40 millisecond
    # hop_size = 0.02    # 20 milliseconds, usually half the window size
    n_fft = int(
        window_size * sample_rate
    )  # Convert window size to number of sampling points
    # hop_length Defines the degree of overlap between windows, affecting the resolution of the spectrogram on the time axis
    hop_length = int(hop_size * sample_rate)  # Convert frame shift to sample count

    # Calculate Mel spectrum
    # n_mels = 40 # The number of Mel filters

    # setup Mel Spectrogram transformer
    mel_transformer = MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        f_min=0,
        f_max=16000,
    )

    melspec = mel_transformer(audio)

    return melspec


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
    device = specgram.device  # Get the device of the input tensor
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
            coeffs = self.beta_distribution.rsample(torch.Size((bs,)))

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

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

from common.audiotransform import rating_value_interplote, audio_weight

from common.audioprocess import (
    CustomCompose,
    CustomOneOf,
    NoiseInjection,
    GaussianNoise,
    PinkNoise,
    AddGaussianNoise,
    AddGaussianSNR,
    read_audio,
)


class BirdclefDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        bird_category_dir: str,
        audio_dir: str = "../../data/train_audio",
        train: bool = True,
    ):
        """
        parameters:
            df: the dataframe of metadata (train/val)
            bird_category_dir: the directory of the bird category array file (npy)
            audio_dir: the parent path where all audio files stored
            train: If the Datset for train set or val set
        """
        super().__init__()
        # if the Dataset for training or validation
        self.train = train
        self.raw_df = df

        # inperplote nan or 0 value of rating col
        self.raw_df = rating_value_interplote(df=self.raw_df)
        # calculate the weight for each audio file through feature: rating
        self.raw_df = audio_weight(self.raw_df)

        self.audio_dir = audio_dir

        self.bird_cate_array = np.load(bird_category_dir, allow_pickle=True)

        self.np_audio_transforms = (
            self.setup_transforms()
        )  # initialize data augmentation func

    def setup_transforms(self):

        return CustomCompose(
            [
                CustomOneOf(
                    [
                        NoiseInjection(p=1, max_noise_level=0.04),
                        GaussianNoise(p=1, min_snr=5, max_snr=20),
                        PinkNoise(p=1, min_snr=5, max_snr=20),
                        AddGaussianNoise(
                            min_amplitude=0.0001, max_amplitude=0.03, p=0.5
                        ),
                        AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=15, p=0.5),
                    ],
                    p=0.3,
                    # It will only choose and apply a transformation 30% of the time.
                    # This means that 70% of the time, no transformation will be applied and the audio sample will remain as is.
                    # If you apply this combined transformation multiple times and choose to do nothing most of the time,
                    # you will observe that the data appears to be unchanged.
                ),
            ]
        )

    def get_audio_path(self, file_name: str) -> str:
        """
        grab the audio file path corresponding to the variable: index in the provided train metadata csv file
        this func would only return one path, because only provide one index

        Parameters:
            file_name: in format category_type/XC-ID.ogg (asbfly/XC134896.ogg)

        Return:
            the single audio path string
        """

        # concatenate parent path and child path
        return os.path.join(self.audio_dir, file_name)

    def target_clip(
        self, index: int, audio: torch.Tensor, sample_rate: int
    ) -> torch.Tensor:
        """
        calculate the index corresponding audio clip

        information from the train metadata csv

        Parameters:
            audio: the raw audio in tensor [num_channels,length]
            sample_rate: audio sampling rate
        """
        # get the strat time of audio clip based off index
        clip_start_time = self.raw_df["clip_start_time"].iloc[index]
        duration_seconds = self.raw_df["duration"].iloc[index]

        # define clip length
        segment_duration = 5 * sample_rate

        # Total number of samples in the waveform
        total_samples = audio.shape[1]

        if clip_start_time <= duration_seconds:
            clip_start_point = clip_start_time * sample_rate
            # For the last clip, the original audio may not be long enough, so we need to use a mask to fill the sequence.
            # The first step is to confirm whether the clip duration is sufficient
            # do not need add mask if the clip duration is enough
            if clip_start_point + segment_duration <= total_samples:
                clip = audio[:, clip_start_point : clip_start_point + segment_duration]

            # add mask if not enough
            else:
                padding_length = clip_start_point + segment_duration - total_samples
                silence = torch.zeros(audio.shape[0], padding_length)
                # concat the last part of the raw audio with silence clip
                clip = torch.cat((audio[:, clip_start_point:], silence), dim=1)

                del silence, padding_length

        else:
            raise ValueError("The clip start time is out of raw audio length")

        del clip_start_time, segment_duration, total_samples

        return clip

    def random_audio_augmentation(self, audio: torch.Tensor):
        """
        audio (torch.Tensor): A 2D tensor of audio samples with shape (1, N), where N is the number of samples.
        """

        audio_aug = self.np_audio_transforms(audio[0].numpy())

        # tranfer the array to 2D tensor and keep the num channel is 1
        # this step is to keep the input and output shape adn type are the same

        audio_aug_tensor = torch.from_numpy(audio_aug)
        audio_aug_tensor = audio_aug_tensor.unsqueeze(0).to(dtype=torch.float16)

        del audio_aug

        return audio_aug_tensor

    def audio_label_tensor_generator(self, true_label: str) -> torch.Tensor:
        """
        Generate a tensor containing all categories based on the given real audio label

        Parameters:
            true lable: a label string

        Return:
            If have 10 class, and give a true lable
            the return should be tensor([0,1,0,0,0,0,0,0,0,0])
        """
        # find the target index in the array
        idx = np.where(self.bird_cate_array == true_label)[0][0]

        # create a zero tensor, the tensor length equals to the array
        audio_label_tensor = torch.zeros(len(self.bird_cate_array), dtype=torch.float16)

        # set the corresponding index value as 1
        audio_label_tensor[idx] = 1

        return audio_label_tensor

    def __len__(self):
        return self.raw_df.shape[0]

    def __getitem__(self, index):
        row = self.raw_df.iloc[index]

        audio_label = row["primary_label"]
        audio_weight = row["audio_weight"]

        # grab the single audio file path
        single_audio_dir = self.get_audio_path(row["filename"])

        # read audio array based off the path
        audio, sr = read_audio(single_audio_dir)

        # augmentation
        # only used for train df
        # do not do augmentation if for validation set
        if self.train:
            audio_augmentation = self.random_audio_augmentation(audio=audio)
            # get the corresponding audio clip based off the index
            clip = self.target_clip(index, audio=audio_augmentation, sample_rate=sr)
            del audio_augmentation
        else:
            clip = self.target_clip(index, audio=audio, sample_rate=sr)

        # change audio label to one-hot tensor
        audio_label_tensor = self.audio_label_tensor_generator(true_label=audio_label)

        audio_label_tensor = torch.tensor(audio_label_tensor, dtype=torch.float16)
        clip = torch.tensor(clip, dtype=torch.float16)
        audio_weight = torch.tensor(audio_weight, dtype=torch.float16)

        del audio

        return audio_label_tensor, clip, audio_weight

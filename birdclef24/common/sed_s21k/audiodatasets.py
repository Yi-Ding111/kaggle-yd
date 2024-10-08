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

from common.sed_s21k.audiotransform import rating_value_interplote,audio_weight

from common.sed_s21k.audioprocess import CustomCompose,CustomOneOf,NoiseInjection,GaussianNoise,PinkNoise,AddGaussianNoise,AddGaussianSNR,read_audio




class BirdclefDataset(Dataset):
    def __init__(self,df:pd.DataFrame,bird_category_dir:str,audio_dir:str='../../data/train_audio',train:bool=True):
        '''
        parameters:
            df: the dataframe of metadata (train/val)
            bird_category_dir: the directory of the bird category array file (npy)
            audio_dir: the parent path where all audio files stored
            train: If the Datset for train set or val set
        '''
        super().__init__()
        # if the Dataset for training or validation
        self.train=train
        self.raw_df=df

        # inperplote nan or 0 value of rating col
        self.raw_df=rating_value_interplote(df=self.raw_df)
        # Calculate the weight of each audio file by rating
        self.raw_df=audio_weight(self.raw_df)

        self.audio_dir=audio_dir

        self.bird_cate_array=np.load(bird_category_dir,allow_pickle=True)

    def get_audio_path(self,file_name:str) -> str:
        '''
        Get the audio path of the corresponding index through the provided train metadata csv file. 
        Since there is only one index, only one path will be returned.

        Parameters:
            file_name: in format category_type/XC-ID.ogg (asbfly/XC134896.ogg)

        Return:
            the single audio path string
        '''

        # concatenate parent path and child path
        return os.path.join(self.audio_dir,file_name)


    def target_clip(self,index:int,audio:torch.Tensor,sample_rate:int)->torch.Tensor:
        """
        calculate the index corresponding audio clip 

        information from the train metadata csv

        Parameters:
            audio: the raw audio in tensor [num_channels,length]
            sample_rate: audio sampling rate
        """
        # Get the audio start time corresponding to index
        clip_start_time=self.raw_df['clip_start_time'].iloc[index]
        duration_seconds=self.raw_df['duration'].iloc[index]

        # define clip length
        segment_duration = 5 * sample_rate

        # Total number of samples in the waveform
        total_samples = audio.shape[1]

        if clip_start_time<=duration_seconds:
            clip_start_point=clip_start_time*sample_rate
            # For the last clip, the original audio may not be long enough, so we need to use a mask to fill the sequence
            # The first step is to confirm whether the length is sufficient
            # The length is sufficient, no mask is needed
            if clip_start_point+segment_duration<=total_samples:
                clip=audio[:, clip_start_point:clip_start_point + segment_duration]

            # need masks if length not enough
            else:
                padding_length = clip_start_point+segment_duration - total_samples
                silence = torch.zeros(audio.shape[0], padding_length)
                # concat the last segment of raw audio with silence
                clip=torch.cat((audio[:,clip_start_point:],silence),dim=1)

                del silence,padding_length
                
        else:
            raise ValueError('The clip start time is out of raw audio length')
        
        del clip_start_time, segment_duration, total_samples

        return clip


    def random_audio_augmentation(self,audio:torch.Tensor):
        '''
        audio (torch.Tensor): A 2D tensor of audio samples with shape (1, N), where N is the number of samples.
        '''
        np_audio_transforms = CustomCompose(
            [
                CustomOneOf(
                    [
                        NoiseInjection(p=1, max_noise_level=0.04),
                        GaussianNoise(p=1, min_snr=5, max_snr=20),
                        PinkNoise(p=1, min_snr=5, max_snr=20),
                        AddGaussianNoise(min_amplitude=0.0001, max_amplitude=0.03, p=0.5),
                        AddGaussianSNR(min_snr_in_db=5, max_snr_in_db=15, p=0.5),
                    ],
                    p=0.3,  
                ),
            ]
        )

        audio_aug=np_audio_transforms(audio[0].numpy())

        # tranfer the array to 2D tensor and keep the num channel is 1
        # this step is to keep the input and output shape adn type are the same

        audio_aug_tensor = torch.from_numpy(audio_aug).unsqueeze(0).to(dtype=torch.float16)

        del audio_aug

        return audio_aug_tensor
    

    def audio_label_tensor_generator(self,true_label:str)-> torch.Tensor:
        '''
        Generate a tensor containing all categories based on the given real audio label

        Parameters:
            true lable: a label string

        Return:
            If have 10 class, and give a true lable
            the return should be tensor([0,1,0,0,0,0,0,0,0,0])
        '''
        # Find the index of the target value in the array
        idx = np.where(self.bird_cate_array == true_label)[0][0]
        
        # Create a tensor of all zeros, with length equal to the length of the array
        audio_label_tensor = torch.zeros(len(self.bird_cate_array), dtype=torch.float16)

        # Set the value of the corresponding index position to 1
        audio_label_tensor[idx] = 1

        return audio_label_tensor


    
    def __len__(self):
        return self.raw_df.shape[0]

    def __getitem__(self,index):
        row=self.raw_df.iloc[index]

        audio_label=row['primary_label']
        audio_weight=row['audio_weight']

        # Get the path of a single audio file
        single_audio_dir=self.get_audio_path(row['filename'])

        # Read the audio array according to the path
        audio, sr=read_audio(single_audio_dir)

        # augmentation
        if self.train:
            audio_augmentation=self.random_audio_augmentation(audio=audio)
            # Get the audio clip corresponding to index
            clip=self.target_clip(index,audio=audio_augmentation,sample_rate=sr)
            del audio_augmentation
        else:
            clip=self.target_clip(index,audio=audio,sample_rate=sr)

        # change audio label to one-hot tensor
        audio_label_tensor=self.audio_label_tensor_generator(true_label=audio_label)

        audio_label_tensor=torch.tensor(audio_label_tensor, dtype=torch.float16)
        clip=torch.tensor(clip, dtype=torch.float16)
        audio_weight=torch.tensor(audio_weight, dtype=torch.float16)

        del audio

        return audio_label_tensor,clip,audio_weight
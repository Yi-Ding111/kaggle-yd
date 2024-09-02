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



# Need to interpolate missing values ​​for ratings in metadata csv file

def rating_value_interplote(df:pd.DataFrame):
    '''
    interplote Nan values for rating col in metadata csv 

    parameters:
        df: the df of the metadata csv file

    rating col means the quality of the corresponding audio file
        5 is high quality
        1 is low quality
        0 is without defined quality level
    '''

    if df['rating'].isna().sum()>0: # with missing values
        df['rating'].fillna(0, inplace=True)

    # Random assign a value to all places where the value is 0, and select from the specified choices
    mask = df['rating'] == 0  # Create a boolean mask indicating which positions are 0

    choices=np.arange(0.5,5.1,0.5).tolist() # [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    random_values = np.random.choice(choices, size=mask.sum())  # Generate random numbers for these 0 values  
    df.loc[mask, 'rating'] = random_values  # Fill the generated random numbers back into the corresponding positions of the original DataFrame

    return df



# Calculate the weight of each audio file through rating, which is helpful for model training
def audio_weight(df):
    '''
    calculate the weight corresponding to each audio file through the rating value

    Because each audio has different quality level, we use weight to affect the inportance of each audio in models,
    the lower the quality of the audio, the lower the weight
    '''
    # Through rating, we calculate the credibility of each audio and express it through weight. 
    # The purpose of this is to improve the model by increasing the weight of high-quality audio and reducing the weight of low-quality audio.
    df["audio_weight"] = np.clip(df["rating"] / df["rating"].max(), 0.1, 1.0)

    return df



# Because this is an unbalanced dataset, the amount of data in each category is very different
# So I will calculate the weight of each category here
# **(-0.5) The purpose is to reduce the relative influence of high-frequency categories and increase the influence of low-frequency categories, 
# so as to help the model better learn those uncommon categories
# The purpose of calculating this is to build a WeightedRandomSampler, 
# so that each time a batch is extracted using dataloader, it is more friendly to data of different categories.

def sampling_weight(df)->torch.Tensor:
    '''
    calculate the sampling weight of each audio file

    because this is imbalanced dataset
    we hope the category with less data has large probability to be picked.
    '''
    sample_weights = (df['primary_label'].value_counts() / df['primary_label'].value_counts().sum()) ** (-0.5)

    # Map weights to each row of the original data
    sample_weights_map = df['primary_label'].map(sample_weights)

    # Convert pandas Series to NumPy array
    sample_weights_np = sample_weights_map.to_numpy(dtype=np.float32)

    # Convert a NumPy array to a PyTorch tensor using torch.from_numpy
    sample_weights_tensor = torch.from_numpy(sample_weights_np)

    return sample_weights_tensor




def dataloader_sampler_generate(df):
    '''
    prepare sampler for dataloader
    '''
    sample_weights_tensor=sampling_weight(df=df)
    # Here we will build an argument sampler that will be used by the dataloader
    # It should be noted that the order of weights in the constructed sampler needs to be consistent with the order of data passed into the dataloader, 
    # otherwise the weights will not match

    # Create a sampler based on the newly obtained weight list
    sampler = WeightedRandomSampler(sample_weights_tensor.type('torch.DoubleTensor'), len(sample_weights_tensor),replacement=True)

    return sampler



def class_weight_generate(df:pd.DataFrame,loaded_array:np.array)->torch.Tensor:
    '''
    Then use focal loss, you need to provide the weight of each category to handle unbalanced data sets

    Parameters:
        loaded_array: an array inlcudes all classes with fixed order.
    '''
    sample_weights = (df['primary_label'].value_counts() / df['primary_label'].value_counts().sum()) ** (-0.5)

    # Convert sample_weights to a DataFrame for easier processing
    sample_weights_df = sample_weights.reset_index()
    sample_weights_df.columns = ['label', 'weight']

    # Convert loaded_array to Categorical type and sort sample_weights_df according to this new order
    sample_weights_df['label'] = pd.Categorical(sample_weights_df['label'], categories=loaded_array, ordered=True)


    ## Sort the DataFrame according to the new category order
    sample_weights_df = sample_weights_df.sort_values('label').reset_index(drop=True)

    class_weight=torch.tensor(sample_weights_df['weight'].values,dtype=torch.float16)
    
    return class_weight
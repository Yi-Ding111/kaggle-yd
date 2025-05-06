from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import h5py
import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import torchvision.transforms as transforms
from PIL import Image
import random
import matplotlib.pyplot as plt
import logging
from sklearn.model_selection import train_test_split
import mlflow

sys.path.append("../donut/src/test/")
from common.config import cfg

from image_recog_datasets import ImageClsDataset
from class_weighting import compute_class_weights


class ImageClsDataModule(L.LightningDataModule):
    def __init__(
        self,
        hdf5_file: str,
        batch_size:int=32,
        train_ratio: float=0.8,
        ds_total_len:float=2.0,
        initial_augment_prob:float=0.8,
        final_augment_prob:float=0.3,
        num_epochs:int=50,
        image_height:int=224,
        image_width:int=224,
        device: torch.device = torch.device("cpu"),
        num_workers: int = 0,
    ):
        """
        Params:
            hdf5_file: the path of the hdf5 dataset
            batch_size: size of batch
            train_ratio: train/val dataset ratio
            ds_total_len: Define how many times the amount of data (data augmentation + original data) the dataset will process is the original dataset
                e.g.: ds_total_len=2 represents that this dataset would process 2 times the amount of images than the original dataset size.
                    In other words, to process all original images and the same number of augmented images as original images.
            initial_augment_prob: Initial data augmentation ratio
            final_augment_prob: final data augmentation ratio
            num_epochs: the number of epochs
            device: USE GPU/CPU deivce
            num_worker: the number of workers for dataloader
        """
        super().__init__()
        self.hdf5_file = hdf5_file
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.ds_total_len = ds_total_len
        self.initial_augment_prob = initial_augment_prob
        self.final_augment_prob = final_augment_prob
        self.num_epochs = num_epochs
        self.device = device
        self.num_workers = num_workers
        self.image_height=image_height
        self.image_width=image_width

        self.current_epoch = 0

    def setup(self, stage=None):
        """
        Split dataset into train and testsets
        """

        with h5py.File(self.hdf5_file, "r") as hf:
            labels = np.array(hf["labels"])
            num_samples = len(labels)

        # ensure the label balancing
        train_indices, val_indices = train_test_split(
            np.arange(num_samples),
            test_size=1 - self.train_ratio,
            stratify=labels,
            random_state=42,
        )

        # create train and val dataset
        self.train_dataset = ImageClsDataset(
            hdf5_file=self.hdf5_file,
            indices=train_indices,
            augment=True,
            ds_total_len=self.ds_total_len,
            device=self.device,
            image_height=self.image_height,
            image_width=self.image_width
        )

        self.val_dataset = ImageClsDataset(
            hdf5_file=self.hdf5_file,
            indices=val_indices,
            augment=False,
            # hard coded as 1 because onlu use original images in val set
            ds_total_len=1,
            device=self.device,
            image_height=self.image_height,
            image_width=self.image_width
        )

    def update_augment_prob(self):
        """
        update the augmented ratio for each epoch

        use more augmented images in early-stage epoches.
        use fewer augmented images in late-stage epoches.
        """
        # sync the current training epoch
        if self.trainer is not None:
            self.current_epoch = self.trainer.current_epoch

        self.augment_prob = self.initial_augment_prob - (
            self.current_epoch / self.num_epochs
        ) * (self.initial_augment_prob - self.final_augment_prob)

        # print(f"📢 Epoch {self.current_epoch}: Augment Prob = {self.augment_prob:.2f}")
        # logging.info(
        #     f"📢 Epoch {self.current_epoch}: Augment Prob = {self.augment_prob:.2f}"
        # )

        return None

    def train_dataloader(self):
        """
        define how the dataloader created for each epoch
        requirements:
            set `reload_dataloaders_every_n_epochs=1` in L.trainer
        """
        # calculate the current epoch's augment ratio
        self.update_augment_prob()
        # self.train_dataset.update_augment_prob(self.augment_prob)

        # re-build the sampler
        sample_weights = compute_class_weights(
            labels=self.train_dataset.labels,
            ds_total_len=self.ds_total_len,
            augment_prob=self.augment_prob,
        )

        train_sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        # print(
        #     f"TRAIN_DATALODER DATAMODULE FLAG (current epoch num): {self.current_epoch+1}"
        # )

        # logging.info(
        #     f"TRAIN_DATALODER DATAMODULE FLAG (current epoch num): {self.current_epoch+1}"
        # )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        define how the validation dataloader created for each epoch
        requirements:
            set `reload_dataloaders_every_n_epochs=1` in L.trainer
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    # def count_train_augmented_original_ratio(self):
    #     """
    #     TEST

    #     check the data distribution in each dataloader
    #     : if the number of original data and augmented data could fit with the expection
    #     """
    #     length = self.train_dataset.base_length
    #     dataloader = self.train_dataloader()

    #     for images, labels, indices in dataloader:
    #         print(
    #             f"TRAIN: If original images:{['original' if idx < length else 'augmented' for idx in indices]}"
    #         )
    #         logging.info(
    #             f"TRAIN: If original images:{['original' if idx < length else 'augmented' for idx in indices]}"
    #         )
    #         print(
    #             f"TRAIN: augmented/original ratio: {Counter(['original' if idx < length else 'augmented' for idx in indices])}"
    #         )
    #         logging.info(
    #             f"TRAIN: augmented/original ratio: {Counter(['original' if idx < length else 'augmented' for idx in indices])}"
    #         )
    #         break

    #     print(
    #         f"TRAIN: Epoch {self.current_epoch+1}/{self.num_epochs} - Augment Prob: {self.augment_prob:.2f}"
    #     )
    #     logging.info(
    #         f"TRAIN: Epoch {self.current_epoch+1}/{self.num_epochs} - Augment Prob: {self.augment_prob:.2f}"
    #     )

    #     return None

    # def count_val_augmented_original_ratio(self):
    #     """
    #     TEST

    #     check the data distribution in each dataloader
    #     : if the number of original data and augmented data could fit with the expection
    #     """
    #     length = self.val_dataset.base_length
    #     dataloader = self.val_dataloader()

    #     for images, labels, indices in dataloader:
    #         print(
    #             f"VALIDATION: If original images:{['original' if idx < length else 'augmented' for idx in indices]}"
    #         )
    #         logging.info(
    #             f"VALIDATION: If original images:{['original' if idx < length else 'augmented' for idx in indices]}"
    #         )

    #         print(
    #             f"VALIDATION: augmented/original ratio{Counter(['original' if idx < length else 'augmented' for idx in indices])}"
    #         )
    #         logging.info(f"VALIDATION: augmented/original ratio: {Counter(['original' if idx < length else 'augmented' for idx in indices])}"
    #         )
    #         break

    #     return None

    # def on_train_epoch_start(self):
    #     """
    #     sync the self.current_epoch value with the trainer
    #     """
    #     if self.trainer is not None:
    #         self.current_epoch = (
    #             self.trainer.current_epoch
    #         )  # sync the current training epoch
    #         self.update_augment_prob()

    #         print(f"\n🔥 Epoch {self.current_epoch+1}/{self.num_epochs}")
    #         print(f"📢 Augment Prob = {self.augment_prob:.2f}")

    #     return None
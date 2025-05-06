from torch.utils.data import Dataset
import h5py
import torch
import numpy as np
import sys
import torchvision.transforms as transforms
from PIL import Image
import random

sys.path.append("../donut/src/test/")
from common.config import cfg


class ImageClsDataset(Dataset):
    def __init__(
        self,
        hdf5_file: str,
        augment: bool = False,
        indices: np.ndarray = None,
        ds_total_len: float = 2,
        image_height: int = 224,
        image_width: int = 224,
        device: torch.device = torch.device("cpu"),
    ):
        """
        hdf5_file: the path of the hdf5 dataset
        augment: choose if doing augmentation on image or not
        indices: used to split the train/val dataset
        ds_total_len: Define how many times the amount of data (data augmentation + original data) the dataset will process is the original dataset
            e.g.: ds_total_len=2 represents that this dataset would process 2 times the amount of images than the original dataset size.
                  In other words, to process all original images and the same number of augmented images as original images.
        image_height | image_width: the size of the image as the input into the model
        device: the CPU/GPU device
        """
        self.hdf5_file = hdf5_file
        self.augment = augment
        self.indices = indices
        self.ds_total_len = ds_total_len
        self.image_height = image_height
        self.image_width = image_width
        self.device = device

        with h5py.File(self.hdf5_file, "r") as hf:
            self.base_length = hf["images"].shape[0]
            # load all labels into memory at one time (high efficiency)
            self.labels = hf["labels"][:]

        # if without given indices, do not do any operations on dataset
        if self.indices is None:
            self.indices = np.arange(len(self.labels))

        # if with indices, update the labels and dataset length
        self.labels = self.labels[self.indices]
        self.base_length = len(self.indices)

        # data augmentation
        self.augmentation_transforms = {
            "Horizontal Flip": transforms.RandomHorizontalFlip(p=1.0),
            "Rotation": transforms.RandomRotation(degrees=10),
            "Color Jitter": transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            ),
            "Random Erasing": transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=1.0, scale=(0.02, 0.33), value=0),
                    transforms.ToPILImage(),
                ]
            ),
        }

        # resizing original images
        self.base_transform = transforms.Resize((self.image_height, self.image_width))

        # normalization(fit for ImageNet pretraining)
        self.normalize_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # process all original images and the same number of augmented images as original images
        self.length = self.ds_total_len * self.base_length

    def random_resized_crop(self, img):
        """
        Crop the image randomly
        """
        crop_transform = transforms.RandomResizedCrop(
            self.image_width, scale=(0.3, 1.0)
        )

        return crop_transform(img)

    def center_included_random_crop(self, img):
        """
        Ensure random cropping within the center point
        """
        W, H = img.size
        center_x, center_y = (
            W // 2,
            H // 2,
        )  # calculate the center point of the original image

        scale_min, scale_max = 0.3, 0.6  # allowable crop ratio
        crop_ratio = random.uniform(scale_min, scale_max)  # generate the crop ratio
        crop_W, crop_H = int(W * crop_ratio), int(H * crop_ratio)

        # make sure the original image's center point in the crop frame
        left_min = max(0, center_x - crop_W)
        top_min = max(0, center_y - crop_H)
        left_max = min(W - crop_W, center_x)
        top_max = min(H - crop_H, center_y)

        # random select the crop frame's coordinates
        left = random.randint(left_min, left_max)
        top = random.randint(top_min, top_max)
        right = left + crop_W
        bottom = top + crop_H

        # resizing
        cropped_img = img.crop((left, top, right, bottom))
        resize_transform = transforms.Resize((self.image_height, self.image_width))

        return resize_transform(cropped_img)

    def apply_augmentations(self, img):
        """
        Apply random augmentation to the image
        """

        # random select the cropping method
        crop_methods = [self.random_resized_crop, self.center_included_random_crop]
        img = random.choice(crop_methods)(img)

        # random select other augmentation methods
        num_augmentations = random.randint(1, len(self.augmentation_transforms))
        chosen_augmentations = random.sample(
            list(self.augmentation_transforms.keys()), num_augmentations
        )

        for augmentation in chosen_augmentations:
            img = self.augmentation_transforms[augmentation](img)

        return img

    def __len__(self):
        """
        return the number of data in the dataset
        """
        return self.length

    def __getitem__(self, idx):
        # judge if need do data augmentation (bool)
        is_augmented = idx >= self.base_length

        # if need data augmentation, need to map the idx into original dataset (subtract base_length)
        # and use the original image corresponding to idx for data augmentation to generate the augmented image
        true_idx = idx if not is_augmented else int(idx - int(self.base_length))

        with h5py.File(self.hdf5_file, "r") as hf:
            image = hf["images"][self.indices[true_idx]]
            label = self.labels[true_idx]

        # convert NumPy -> PIL（fit with torchvision.transforms）
        image = Image.fromarray(image)

        # resizing first
        image = self.base_transform(image)

        # data augmentation
        if is_augmented and self.augment:
            image = self.apply_augmentations(image)

        # DO normalization, we use pretained weights, so follow this normalization
        image = self.normalize_transform(image)

        # return image.to(self.device), torch.tensor(
        #     label, dtype=torch.long, device=self.device
        # )

        return image, torch.tensor(label, dtype=torch.long)

# Image preprocessing


## EDA

1. the image types are imbalanced
2. all annotation ground truth (data-series) follow the same architecture
3. * in vertical_bar, all gt value (x is string, y is int or float)
    * in dot, all x is string, y is int
    * in line, all x is string, y is int or flot
    * in horizontal_bar, all x is int or float, y is str
    * in scatter, all x is float or int, y is float or int

easy to find that each type of graphs has the similar distribution and gt architecture.

However, Here are some problems we need to know:

1. there is a problem that different graphs components is too consistant to do generalization. For one example, in dot, all x labels are string, but in scatter, all x labels are numeric.

2. class imblanced, we need to deal with the imbalanced problem. 

## ImageClean

need to check the information in the corresponding annotation file chould fit with the image file.

Some annotation files includes wrong info, we need to filter them out, instead the training would be errors.

* First I crop the image based of the chart main boody boundary provided by the annotation filres.
* Then padding the image with 0 to same size.
* check how many 0-paddings in the resized images.
* I find 3 images have wrong annotation files.

This is quite small part of dataset. just drop them out directly.


## ImageAnnotationResize

I need to use models like transformers which ask me ensure that all input images' sizes should be fixed. I need to scale the annotation data if i resize images.

Right here i checked the scaling method and make sure that the function could work well.


## ImageResizing

The raw images are in various sizes. To accelerate the model traing. I choose to resize images into the same size (640,640)

scaling the annotations based off the original height, width and new height, width.

I am not doing data augmentation here, but performing this step while reading the data and before feeding into the model for training.
Because I want to keep the data's randomness to improve the model's generalization.


## ImageTransformation

In order to accelerate the data processing when traing the model. I transfer images and labels into HDF5 format.

Because this dataset is very imbalanced, I need to use weighted sampling and other ways to reduce the influence when doing the data processing.

Do not make the data as batches into HDF5 dataset.

## ImageAugmentation

enrich the dataset， would do 

- Random Resized Crop
- Random Resized crop (cover center point in original image in crop frame)
- Random Horizontal Flip
- Random Rotation, ±10°
- Random Erasing
- Color Jittering
- Fine-tuning at Higher Resolution

when doing augmentation like rotating or flipping, we must make sure that the related annotation file changed as well. Because the annotation file's content is highly related with image location (pixel position).

If only doing the classification, do not worry about annotations.


## ImageLightningDataset

organise the torch dataset for model training. Here includes some data transformation and augmentation steps.

- add the data augmentation steps into dataset
- add original data as input, not only using images after augmentation
- Make sure all original images are taken as input and also add the augmented images
- the augmented images are allowed resampling from original images
- **plan using more augmented images in batch in early-stage training, and using more original images at the late-stage, to imporve the model's robust**

I do a fake trainer to test the Dataset and Datamoduling at last.
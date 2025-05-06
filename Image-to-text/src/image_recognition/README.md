# Image recognition

I need to build up a model first for image classification

I build up:

1. image recognition datasets (including transformations, data augmentation. see details in /preprocess)
2. image recognition datasets (including train-test split, weighted sampling, weighted sampling from original and augmented images through dynamic ratios (allow more augmented images in batches in early training stage and more original images in batches in late training stage))
3. model training module (local development and training on kaggle leverageing GPU)
4. model inferencing with ckpt or pt model. 

The final model is [image recognition model](../../../trained_models/vit-best-epoch=12-val_acc=0.9985.pt)


# BirdCLEF 2024


This model uses the SED (Sound Event Detection) framework. ![Sound Event Detection](https://arxiv.org/pdf/2107.05463)


Due to the limitation of the data we obtained, we only have the label of the extrie data (audio). We do not know which time period in the entire audio data is the time when the label appears. Therefore, it is challenging to use strong label annotation here. We choose weak label for data preparation. Weak label means that the same label is used for the entire audio recording. 

In the data process here, the final label tensor format is converted into the following format:


```python

tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0.])

```

This indicates that a combined audio contains multiple objects, and because each object is marked with a weak label, only the active object is marked, and the timestamp is not marked.



* External data


For external datasets, we refer to some previous competitions and filter out some available datasets through the IDs provided in this dataset to enrich the existing data.


* Imbalanced dataset

After adding some external data, the overall data set is still very unbalanced. It should be noted here that:

1. To solve the problem of data imbalance, we enable __inverse frequency weighting__, that is, use $ x^{(-0.5)}$ as the calculation method of category weight, where x can be the number of samples in a certain category. This can effectively increase the relative weight of rare categories, reduce the relative influence of high-frequency categories, and increase the influence of low-frequency categories, thereby helping the model to learn those uncommon categories better.

2. Batch training is needed here, which means that we not only need to optimize and balance the weights of categories in general, but also for a single batch, we need to ensure that the data within a batch is relatively balanced. After obtaining the weight list mentioned above, the function WeightedRandomSampler will be called to build the sampler, and finally called when the dataloader reads the data.

* Dtata preprocessing

1. Slice the audio data, divide the audio
2. complete the clip, align the length of each clip
3. process corresponding required ground truth, audio clip tensor, label and other content.


* Data augmentation

Include adding Gaussian noise, pinknoise, randomly selecting the signal-to-noise ratio (SNR) of the noise within a given minimum and maximum range, injecting random noise into the audio signal, TimeStretch to stretch the audio, and changing the playback speed to normalize the audio samples.

Adjust the audio pitch, combine the audio, mix the audio, randomly mask some of the audio, and further perform data enhancement.


* Data transformation

Include mixup of audio clips and mel spectrogram transform.

Use first-order and second-order difference methods to increase audio channels. (Increase from single channel to 3 channels).


* Pre-trained model and feature extractor

I tried to manually create the entire neural network for model prediction, but the output was poor, so I introduced a pre-trained model as a feature extractor. Here I chose the EfficientNet series. The last part of the layer of this model, the architecture frozen parameters are retained, and used as a feature extractor.

After that, ChronoNet and attention block will be added to train the model on the high-dimensional features obtained by the feature extractor.



* Training acceleration

Enable __mixed precision__ and __gradient accumulation__ to accelerate training.


* Loss function

Because the dataset is very unbalanced, I use Focal loss as the loss function



---
---

```sh
.
├── README.md
├── aws
│   ├── v1
│   └── v2
├── common
│   ├── sed_s21k
│   └── sed_s21k_v4
├── data
│   ├── predict
│   ├── preprepared
│   ├── previous-years-labeled-file
│   ├── test
│   ├── train
│   ├── train_audio
│   ├── train_metadata.csv
│   ├── train_metadata_new.csv
│   ├── train_metadata_new_add_rating.csv
│   ├── train_metadata_new_add_rating_2500.csv
│   └── train_metadata_new_tiny.csv
├── dev
│   ├── EDA
│   ├── extra-data-collect
│   ├── feature-extractor
│   ├── model-training
│   ├── other
│   └── preprocessing
├── env
├── kaggle-version
├── models
├── notes
├── pkg
└── requirements.txt

```

* ENV

The conda env i used is stored in the folder: env. you could use this env to avoid the version conflicts.

* Kaggle version

All files needed for the kaggle is stored in the folder: kaggel-version except the trained model file.

* Data

Include several files as format examples.

* DEV

All other folders are about the code I built during the development, including some necessary notes for myself to remember some points.




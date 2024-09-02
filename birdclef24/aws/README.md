Files under the folder **aws** are used for the model running in AWS sagemaker.

Because I tried to rent the server on AWS to speed up the model training, so I do some changes for the code.

Anyone who only want to focus on the model code, please ignore this part.

The main part of code is similar, but add some boto3 for the code could run in sagemaker domain. 

I stored data in s3 bucket, and read the data into sagemaker domain through boto3. 

I dropped this approach at the end, because I found it is easy to run if I store pre-trained data in somewhere for the model training.


## General:

1. Store data in s3 bucket
2. read data into sagemaker domain through boto3
3. run notebook for model training


### v1

* audiodatasets.py:

    - do preprocess and calculation on audio clips.
    - get needed ground truth label in ideal format. 
    - transform audio clip as tensor.
    - feed data to model for training.


* audioprocess.py:

    - Include code for data augmentation.
    - Add Guassin noise, pinknoise, SNR, add random noise into audio, timestretech for audio. change audio speed, normalization, etc.

* audiotransfrom.py:

    - Include steps for data transformation.
    - mix up clips, mel spectrogram transform,etc.


* sed_s21.ipynb:

    - this notebook includes all code, I do not split code into packages, so this notebook can not use torch workers (can not define worker numbers). It would affect the speed of model training.

* sed_s21_random_select_sub_set.ipynb

    - almost the same with sed_s21.ipynb

* efficient-in21k-model-add-workers.ipynb

    - code for model training
    - the dataset here is quite imbalanced, I used some approaches to mitigate the bias that the imbalanced dataset brings to the model.
    - Ensure data for each category has samples is extracted into validation set, and the other part join into train set. Then random select part of data from train set, and then add this selected part into validation set.
    - Because of imbalanced dataset, I calculated the weight of each category. I add index calculation (-0.5) to reduce the relative influence of high frequency categories and increase that of the low-frequency categories. This result of weights would be used in dataloader WeightRandomSampler.
    - Invoke the data augmentation module to adjust the audio pitch, combine the audio, mix the audio, add noise, stretch the audio, random mask part of the audio, use the first-order and second-order differences to increase the audio channel, etc.
    - Freeze the parameters of the retained layers of the used pretrained model(efficientnet-21k series).
    - Here, the retained layer of the training model can be regarded as a feature extractor
    - Discard the last two layers and reconstruct the model to match the output format we need, add self-attention blocks and some common layers like: relu, pooling layer, to replace the last two layers of the original model.
    - In order to speed up training, we choose to use gradient accumulation and mixed precision for training calculations.


### v2

* audiodatasets.py: 

    - Include torch dataset class
    - The dataset class includes steps such as reading data, data augmentation, audio segmentation, preparing label data, calculating the weight of individual data, etc.
    - Collate_fn is added so that the data obtained by ModelModule is fully prepared and does not require secondary processing:
        - Do mel spectrogram transformation
        - Normalization
        - random mask part of audio clip
        - Merge multiple audios, update the label list (SED).
        - Add the second-order difference and the first-order difference to divide the single channel into three channels.
    
* audioprocess.py:

    - Interpolate missing values ​​in metadata file (reting).
    - Calculate the weight of each audio data (determined by the audio's credibility/quality score)
    - Prepare sampler, because it is an unbalanced data set, so the sampling will be relatively balanced, used for dataloader.
    - Calculate the weight of each category (unbalanced data set) to prepare for the subsequent calculation of focal loss.

* audiotransform.py:
    
    - Contain data augmentation and transformation functions.

* modelmeasurements.py:

    - Include loss functions for model training.

* modulize-efficient-in21k-change-last-layers.ipynb:

    - compare with the last version:
    - add collate_fn in dataloader
    - update mdoel frame, add part chronoNet model as a part of the model.

This folder contains all the code and process of developing this model

### feature-extractor

* 1-efficientnetv2_2_in21k.ipynb

    - The purpose of this notebook is to verify the input and output of the pre-trained model efficientnetv2_2_in21k.

* 2-tf_efficientnet_b0.ipynb

    - This is to verify the effect of the model tf_efficientnet_b0

### preprocessing

* 1-read_ogg_data_and_fft.ipynb

    - Read the ogg file and try to use fft conversion to try whether this conversion is effective for model training

* 2-compare_various_type_after_fft.ipynb

    - Compare the FFT results of audio data of different categories to check whether there are obvious differences

* 3-compare_same_cate_audio_with_fft.ipynb

    - Compare the FFT results of audio data of two categories

* 4-slice-audio.ipynb

    - Divide the audio and then check the FFT results of each segment of audio data

* 5-5sec_sampling.ipynb

    - This notebook is intended to explore whether it is feasible to randomly sample data for each 5-second window, sample multiple 2-second audios, and view the frequency domain image.

* 6-concat_all_as_tensor_chrononet.ipynb

    - Calculate the data, and finally convert the converted data from time domain to frequency domain, and then calculate the corresponding label

* 7-adjust_sampling_rate.ipynb

    - Change the sampling rate and see the impact

* 9-mel-spectrogram.ipynb

    - Verify mel spectrogram transformation

* 10-MFCC.ipynb

    - Verify MFCC

* 11-torchaudio-update-6.ipynb

    - Use torchaudio package to calculate mel spectrogram

* 11.1-organize-in-torchDataset.ipynb

    - Encapsulate Dataset for subsequent model calls

* 12-predict-data-transform.ipynb

    - Adjust Dataset

* 12.1-predict-data-transform-clean-version.ipynb

    - Simplified version of the previous notebook

* 13-data-process-with-audio-augmentation.ipynb

    - Add data augmentation steps and weight sampling to reduce the impact of unbalanced data sets

* 13.1-data-process-augmentation-with-batch.ipynb

    - This notebook will introduce datasets.Dataset and torch dataloader to process batch data

* 13.2-data-process-augmentation-with-batch-add-mix.ipynb

    - Further add data mixing steps and build the initial model

* 14-re-organize-dataprocess.ipynb

    - This notebook is modified based on the 13/13.1/13.2 notebooks

* 15-re-prepare-data-with-longer-clips.ipynb

    - Change the clip length for training

### model-training

* 1-initial-simple-with-chrononet.ipynb

    - Give chrononet and efficientnet to build the initial model

* 1.2-initial-chrononet-clean-version.ipynb

    - Optimize the previous version code, and keep the model structure unchanged

* 2-chrononet-with-torchDataset-melspecs.ipynb

    - Upgrade the model, add Dataset, mel spectrogram transformation, and enrich the model structure

* 3-efficient-in21k-feature-extractor-pred.ipynb

    - Adjust the model structure and replace the new layer for experiment

* 3.1-efficient-in21k-add-workers.ipynb

    - Packaging function, enable workers to accelerate training

* 4-efficient-in21k-change-last-layers.ipynb

    - Try to move some data processing steps to dataloader instead of lightningModelModule. Use some other layers to replace the attention layer used before

* 4.1-modulize-efficient-in21k-change-last-layers.ipynb

    - According to 4-efficient-in21k-change-last-layers.ipynb, the functions are packaged to facilitate the use of multiple workers to accelerate training

* 4.2-modulize-efficient-in21k-change-last-layers.ipynb

    - This notebook is designed to pre-store the model to disk to facilitate subsequent model reading and training. Store pre-processed data to accelerate model training

* 4.3-efficient-in21k-attention-layer.ipynb

    - Because chrononet layers are used in 4.2 notebook, but the desired effect is not achieved, try to replace it with the attention layer used before

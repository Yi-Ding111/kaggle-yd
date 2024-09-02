#!/bin/bash

set -e


echo "set up..."

brew install libav
brew install ffmpeg
brew install miniforge 

conda env create -f environment.yml



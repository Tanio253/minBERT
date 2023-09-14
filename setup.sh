#!/usr/bin/env bash
conda create -n minbert python=3.11.5
conda activate minbert
conda install pytorch torchvision torchaudio cpuonly -c pytorch
pip install tqdm==4.66.1
pip install requests==2.31.0
pip install importlib-metadata==6.8.0
pip install filelock==3.12.4
pip install sklearn==0.0
pip install tokenizers==0.14.0
pip install explainaboard_client==0.1.4
pip install chardet==5.2.0
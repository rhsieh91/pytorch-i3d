# Scene-Invariant Feature Extractor Network (SIFE-Net) on <Jester, Charades>

## Overview
This repository was forked from https://github.com/piergiaj/pytorch-i3d, which provides a PyTorch I3D model trained on Kinetics. The pre-trained weights are used to finetune a baseline I3D model and to train our network SIFE-NET on 1) a modified version of the 20BN-jester dataset and 2) the Charades dataset.

## Requirements
* pytorch
* torchvision
* tensorboard
* future
* numpy
* pandas
* ffmpeg
* PyAV --> https://github.com/mikeboers/PyAV#installation

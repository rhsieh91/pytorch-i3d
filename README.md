# Scene-Invariant Feature Extractor Network (SIFE-Net) on <Jester, Charades, HVU>

## Overview
SIFE-Net consists of an action recognition network combined with an adversarial branch to reduce bias and achieve scene-invariant video feature learning. The idea is to decouple action and scene features in video understanding and in turn create a more robust action recognition network that truly understands *motion*. Using adversarial learning for debiasing has been well studied in images and has only recently started to be explored for videos. 

The backbone action recognition networks being explored are I3D and 3D-ResNet. For the I3D implementation, this repository was forked from https://github.com/piergiaj/pytorch-i3d, which provides a PyTorch I3D model trained on Kinetics. The pre-trained weights have been used to finetune a baseline I3D model and train SIFE-Net on a modified version of the 20BN-jester dataset and the Charades dataset, 

The 3D-ResNet implementation is in progress and currently being explored on the Holistic Video Understanding (HVU) dataset.

## Requirements
* pytorch
* torchvision
* tensorboard
* future
* numpy
* pandas
* ffmpeg
* PyAV --> https://github.com/mikeboers/PyAV#installation

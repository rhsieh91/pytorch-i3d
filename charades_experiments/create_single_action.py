# Splits Charades RGBs into single-action samples, in place

import cv2
import argparse
import os
import pandas as pd
import numpy as np
import math
from shutil import copyfile

if __name__ == '__main__': 
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--csv_path', type=str) # <relative-path>/Charades_v1_train.csv
    parser.add_argument('--input_root', type=str) # directory containing 24FPS RGBs
    parser.add_argument('--target_root', type=str) # directory to save single-action samples

    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)

    NUM_FPS = 24
    for i, row in df.iterrows():
        # for a single sample
        if type(row['actions']) is not str:
            continue
        groups = row['actions'].split(';')

        frame_names = sorted(os.listdir(os.path.join(args.input_root, row['id'])))
        for j, group in enumerate(groups):
            # this will turn into its own sample
            action, start, end = group.split()
            length = float(end) - float(start)
            num_frames = math.floor(NUM_FPS * length)
            
            # copy from start to end
            for k in range(start * NUM_FPS, end * NUM_FPS):
                src = os.path.join(args.input_root, frame_names[k])
                dst = os.path.join(args.target_root, row['id'] + '-' + str(j).zfill(2))
                if not os.path.exists(dst):
                    os.makedirs(dst)
                copyfile(src, dst)

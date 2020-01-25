# Removes Charades single action samples with time-start > time-end

# import cv2
# import pandas as pd
# import numpy as np
# import math
# import shutil
import argparse
import os

if __name__ == '__main__': 

    parser = argparse.ArgumentParser()    
    parser.add_argument('--rgb_root', type=str) # directory containing 24FPS RGBs
    args = parser.parse_args()

    for subdir, dirs, files in os.walk(args.rgb_root):
        for dir in dirs:
            if not os.listdir(os.path.join(args.rgb_root, dir)):
                os.rmdir(os.path.join(args.rgb_root, dir))
        break
    
    # parser = argparse.ArgumentParser()
    
    # parser.add_argument('--orig_csv_path', type=str) # <relative-path>/Charades_v1_train.csv
    # parser.add_argument('--rgb_root', type=str) # directory containing 24FPS RGBs
    # parser.add_argument('--new_csv_path', type=str) # directory to save single-action samples

    # args = parser.parse_args()

    # # df_orig = pd.read_csv(args.orig_csv_path)

    # for i, row in df_orig.iterrows():
    #     # for a single sample
    #     if type(row['actions']) is not str:
    #         continue
    #     groups = row['actions'].split(';')

    #     frame_names = sorted(os.listdir(os.path.join(args.input_root, row['id'])))
    #     for j, group in enumerate(groups):
    #         # this will turn into its own sample
    #         action, start, end = group.split()

    #         # check if this action label is erroneous (time-start > time-end)
    #         if float(start) >= float(end):
    #             os.rmdir(os.path.join(args.rgb_root, row['id'] + '-' + str(j).zfill(2)))

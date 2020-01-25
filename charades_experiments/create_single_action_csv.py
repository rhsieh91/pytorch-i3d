# Splits Charades RGBs into single-action samples, in place

import argparse
import os
import pandas as pd
import csv

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_input_file', type=str) # <relative-path>/Charades_v1_train.csv
    parser.add_argument('--csv_output_root', type=str) # directory to save new csv file
    args = parser.parse_args()

    df = pd.read_csv(args.csv_input_file)
    NUM_FPS = 24

    with open(os.path.join(args.csv_output_root, 'Charades_single_action_train.csv'), 'w', newline='') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['id', 'action', 'scene', 'objects'])

        for i, row in df.iterrows():
            # for a single sample
            if type(row['actions']) is not str:
                continue
            groups = row['actions'].split(';')
            id = row['id']
            scene = row['scene']
            objects = row['objects']
            for j, group in enumerate(groups):
                # split into new sample
                id_new = id + '-' + str(j).zfill(2)
                action, start, end = group.split()
                # check if this action label is erroneous (time-start > time-end)
                if float(start) < float(end):
                    filewriter.writerow([id_new, action, scene, objects])
